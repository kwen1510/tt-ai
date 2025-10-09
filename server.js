import express from "express";
import dotenv from "dotenv";
import axios from "axios";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import Groq from "groq-sdk";
import OpenAI from "openai";

dotenv.config();

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));

const {
  PORT,
  PASSWORD,
  APPS_SCRIPT_EXEC,
  GROQ_API_KEY,
  OPENAI_API_KEY,
  OPENAI_MODEL
} = process.env;

// --- LLM clients ---
const groq = GROQ_API_KEY ? new Groq({ apiKey: GROQ_API_KEY }) : null;
const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;
const targetModel = OPENAI_MODEL || "gpt-4.1-mini";
const BASE_SYSTEM_INSTRUCTIONS = [
  "You are a helpful assistant for timetable queries.",
  "Use only the supplied timetable JSON.",
  "Assume the results array already contains the exact rows relevant to the user's request; do not drop entries unless they clearly contradict the question. Treat the data as the verified schedules for the requested people.",
  "List every timetable entry individually without merging, averaging, or omitting timeslots.",
  "Always show the weekday (spell out abbreviations), start time, end time, and any other provided fields for each entry.",
  "Order items chronologically by weekday (Monday through Sunday) and start time whenever you present multiple rows.",
  "Using a table is fine if it keeps every entry visible, but ensure each JSON row appears exactly once.",
  "Please reflect the start and end times precisely. It makes a whole world of difference.",
  "Consult the provided title or notes for extra context when helpful, but never invent new data.",
  "If the data is insufficient to answer, explain what is missing and avoid speculation or new assumptions."
].join(" ");

/** Proxy to Google Apps Script (password/API keys are on server) */
app.post("/proxy", async (req, res) => {
  try {
    const { question, mode = "auto" } = req.body;
    if (!APPS_SCRIPT_EXEC) return res.status(500).json({ ok: false, error: "Missing APPS_SCRIPT_EXEC" });
    if (!question) return res.status(400).json({ ok: false, error: "Missing question" });

    const payload = { password: PASSWORD, mode, question };
    const r = await axios.post(APPS_SCRIPT_EXEC, payload, { headers: { "Content-Type": "application/json" } });
    res.json(r.data);
  } catch (err) {
    console.error("Proxy error:", err?.response?.data || err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

/** Summarize/answer from JSON via Groq (stream → aggregate → OpenAI-like shape) */
app.post("/summarize", async (req, res) => {
  try {
    const { question, results, queryType, timetable, title, notes, teachers, clarify } = req.body;
    const collapsed = JSON.stringify(results ?? null);

    if (clarify && clarify.required && clarify.type === "AMBIGUOUS_TEACHER") {
      const summary = formatClarifyMessage(clarify, question);
      return res.json({ choices: [{ message: { content: summary } }] });
    }

    if (String(queryType || "").toUpperCase() === "FULL_TIMETABLE" && timetable) {
      const summary = formatFullTimetable({ question, timetable, title, notes, teachers });
      return res.json({ choices: [{ message: { content: summary || "No timetable produced." } }] });
    }

    const summary =
      openai
        ? await summarizeWithOpenAI({ question, collapsed })
        : groq
        ? await summarizeWithGroq({ question, collapsed })
        : (() => {
            throw new Error("No LLM provider configured. Set OPENAI_API_KEY or GROQ_API_KEY.");
          })();

    res.json({ choices: [{ message: { content: summary || "No summary produced." } }] });
  } catch (e) {
    console.error("Summarize error:", e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

/** Whisper Turbo via Groq SDK (handles Android/Chrome no-extension case) */
app.post("/whisper", upload.single("audio"), async (req, res) => {
  if (!req.file) return res.status(400).json({ ok: false, error: "No audio file uploaded" });

  // Ensure the temp file has a useful extension for the API
  const origName = req.file.originalname || "";
  const origExt = path.extname(origName);
  const mt = req.file.mimetype || "";
  const guessedExt =
    origExt ||
    (mt.includes("ogg") ? ".ogg" : mt.includes("webm") ? ".webm" : mt.includes("mp4") ? ".mp4" : ".m4a");
  const properPath = req.file.path + guessedExt;

  try {
    fs.renameSync(req.file.path, properPath);

    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(properPath),           // docs pattern
      model: "whisper-large-v3-turbo",
      response_format: "verbose_json",
      language: "en"
    });

    const text = (transcription && typeof transcription.text === "string" && transcription.text.trim()) || "";
    if (!text) {
      return res.status(200).json({ ok: false, error: "empty_transcription", raw: transcription });
    }
    res.json({ ok: true, text, segments: transcription.segments || [] });
  } catch (e) {
    console.error("Whisper error:", e.message);
    res.status(500).json({ ok: false, error: e.message });
  } finally {
    try { fs.unlinkSync(properPath); } catch {}
  }
});

app.listen(PORT || 8080, () => {
  console.log(`Server running on :${PORT || 8080}`);
});

async function summarizeWithOpenAI({ question, collapsed }) {
  const completion = await openai.chat.completions.create({
    model: targetModel,
    messages: [
      { role: "system", content: BASE_SYSTEM_INSTRUCTIONS },
      {
        role: "user",
        content: `Answer ONLY from this JSON.\nQuestion: ${question}\nJSON (collapsed): ${collapsed}`
      }
    ],
    temperature: 0.0,
    max_tokens: 2048
  });
  return completion.choices?.[0]?.message?.content?.trim() || "";
}

async function summarizeWithGroq({ question, collapsed }) {
  const stream = await groq.chat.completions.create({
    model: "openai/gpt-oss-120b",
    messages: [
      { role: "system", content: BASE_SYSTEM_INSTRUCTIONS },
      {
        role: "user",
        content: `Answer ONLY from this JSON.\nQuestion: ${question}\nJSON (collapsed): ${collapsed}`
      }
    ],
    temperature: 0.2,
    top_p: 1,
    max_completion_tokens: 1024,
    stream: true
  });

  let full = "";
  for await (const chunk of stream) full += chunk.choices?.[0]?.delta?.content || "";
  return full.trim();
}

function formatFullTimetable({ timetable, title, notes, teachers }) {
  if (!timetable || typeof timetable !== "object") return "No timetable data provided.";
  const dayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const dayNames = {
    Mon: "Monday",
    Tue: "Tuesday",
    Wed: "Wednesday",
    Thu: "Thursday",
    Fri: "Friday",
    Sat: "Saturday",
    Sun: "Sunday"
  };

  const teacherName = timetable.teacher || (Array.isArray(teachers) && teachers[0]) || "Teacher";
  const heading = title || `${teacherName} timetable`;

  const grouped = Array.isArray(timetable.grouped) && timetable.grouped.length
    ? timetable.grouped
    : buildGroupedFromRows(timetable.rows, dayOrder);

  if (!grouped || !grouped.length) return `No timetable entries found for ${teacherName}.`;

  const sections = [`## ${heading}`];
  if (notes) sections.push(`_${notes}_`);

  grouped
    .slice()
    .sort((a, b) => orderIndex(dayOrder, a.Weekday) - orderIndex(dayOrder, b.Weekday))
    .forEach((day) => {
      const slots = coalesceSlots(Array.isArray(day.slots) ? day.slots : []);
      if (!slots.length) return;
      const fullDay = dayNames[day.Weekday] || day.Weekday;
      sections.push(`\n### ${fullDay}`);
      sections.push("| Period | Start | End | Subject | Class | Room |");
      sections.push("| --- | --- | --- | --- | --- | --- |");
      slots.forEach((slot) => {
        const period = coerceText(slot.Period);
        const start = coerceText(slot.Start);
        const end = coerceText(slot.End);
        const subject = coerceText(slot.Subject);
        const klass = coerceText(slot.Class);
        const room = coerceText(slot.Room);
        sections.push(`| ${period} | ${start} | ${end} | ${subject} | ${klass} | ${room} |`);
      });
    });

  return sections.join("\n").trim();
}

function buildGroupedFromRows(rows, dayOrder) {
  if (!Array.isArray(rows) || !rows.length) return [];
  const grouped = {};
  rows.forEach((row) => {
    if (!row || typeof row !== "object") return;
    const day = row.Weekday || "Unknown";
    if (!grouped[day]) grouped[day] = [];
    grouped[day].push({
      Period: row.Period ?? "",
      Start: row.Start ?? "",
      End: row.End ?? "",
      Subject: row.Subject ?? "",
      Class: row.Class ?? "",
      Room: row.Room ?? ""
    });
  });

  return Object.keys(grouped)
    .sort((a, b) => orderIndex(dayOrder, a) - orderIndex(dayOrder, b))
    .map((day) => ({
      Weekday: day,
      slots: grouped[day]
    }));
}

function coerceText(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return String(value);
  const trimmed = String(value).trim();
  return trimmed.length ? trimmed : "—";
}

function orderIndex(order, key) {
  const idx = order.indexOf(key);
  return idx === -1 ? order.length + 1 : idx;
}

function coalesceSlots(slots) {
  if (!Array.isArray(slots) || !slots.length) return [];

  const normalized = slots
    .map((slot) => normalizeSlot(slot))
    .filter((slot) => slot !== null)
    .sort((a, b) => {
      const aMinutes = timeToMinutes(a.Start);
      const bMinutes = timeToMinutes(b.Start);
      if (!Number.isNaN(aMinutes) && !Number.isNaN(bMinutes) && aMinutes !== bMinutes) {
        return aMinutes - bMinutes;
      }
      const aPeriod = parsePeriodNumber(a.Period);
      const bPeriod = parsePeriodNumber(b.Period);
      if (Number.isFinite(aPeriod) && Number.isFinite(bPeriod) && aPeriod !== bPeriod) {
        return aPeriod - bPeriod;
      }
      return 0;
    });

  const merged = [];
  normalized.forEach((slot) => {
    const periodLabel = slot.Period;
    const last = merged[merged.length - 1];
    if (last && sameSlotGroup(last, slot) && slotsTouch(last, slot)) {
      if (slot.End) last.End = slot.End;
      if (!last.Start && slot.Start) last.Start = slot.Start;
      last.periodsRaw.push(periodLabel);
    } else {
      merged.push({
        periodsRaw: [periodLabel],
        Start: slot.Start,
        End: slot.End,
        Subject: slot.Subject,
        Class: slot.Class,
        Room: slot.Room,
        _subjectKey: slot._subjectKey,
        _classKey: slot._classKey,
        _roomKey: slot._roomKey
      });
    }
  });

  return merged.map((item) => ({
    Period: formatPeriodLabel(item.periodsRaw),
    Start: item.Start,
    End: item.End,
    Subject: item.Subject,
    Class: item.Class,
    Room: item.Room
  }));
}

function normalizeSlot(raw) {
  if (!raw || typeof raw !== "object") return null;
  const toTrimmed = (value) => (value === null || value === undefined ? "" : String(value).trim());
  const subject = toTrimmed(raw.Subject);
  const className = toTrimmed(raw.Class);
  const room = toTrimmed(raw.Room);
  const period = toTrimmed(raw.Period);
  const start = toTrimmed(raw.Start);
  const end = toTrimmed(raw.End);

  return {
    Period: period,
    Start: start,
    End: end,
    Subject: subject,
    Class: className,
    Room: room,
    _subjectKey: normalizeKey(subject),
    _classKey: normalizeKey(className),
    _roomKey: normalizeKey(room)
  };
}

function normalizeKey(value) {
  return value ? value.toLowerCase() : "";
}

function sameSlotGroup(current, next) {
  return (
    current._subjectKey === next._subjectKey &&
    current._classKey === next._classKey &&
    current._roomKey === next._roomKey
  );
}

function slotsTouch(current, next) {
  const lastEnd = timeToMinutes(current.End);
  const nextStart = timeToMinutes(next.Start);
  if (!Number.isNaN(lastEnd) && !Number.isNaN(nextStart) && nextStart === lastEnd) {
    return true;
  }

  const lastPeriodRaw = current.periodsRaw?.[current.periodsRaw.length - 1];
  const lastPeriodNum = parsePeriodNumber(lastPeriodRaw);
  const nextPeriodNum = parsePeriodNumber(next.Period);
  if (Number.isFinite(lastPeriodNum) && Number.isFinite(nextPeriodNum)) {
    return nextPeriodNum === lastPeriodNum + 1;
  }
  return false;
}

function timeToMinutes(value) {
  const match = /^(\d{1,2}):(\d{2})$/.exec(value || "");
  if (!match) return NaN;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return NaN;
  return hours * 60 + minutes;
}

function parsePeriodNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : NaN;
}

function formatPeriodLabel(periodsRaw = []) {
  const filtered = periodsRaw
    .map((p) => (p === null || p === undefined ? "" : String(p).trim()))
    .filter(Boolean);
  if (!filtered.length) return "—";
  if (filtered.length === 1) return filtered[0];
  const numbers = filtered.map((p) => parsePeriodNumber(p));
  const sequential = numbers.every((n, idx) => Number.isFinite(n) && (idx === 0 || n === numbers[idx - 1] + 1));
  if (sequential) {
    return `${filtered[0]}-${filtered[filtered.length - 1]}`;
  }
  return filtered.join(", ");
}
function formatClarifyMessage(clarify, question) {
  const candidates = Array.isArray(clarify.candidates) ? clarify.candidates : [];
  const uniqueCandidates = Array.from(new Set(candidates.filter(Boolean)));
  const picked = clarify.input || "";
  const title = clarify.message || "Multiple matches found.";
  if (!uniqueCandidates.length) {
    return [
      `**${title}**`,
      "",
      `“${picked}” matches multiple teachers. Please provide a more specific name so I can continue.`,
      question ? `Original question: _${question}_` : ""
    ].filter(Boolean).join("\n");
  }

  const list = uniqueCandidates.map((name) => `- ${name}`).join("\n");
  return [
    `**${title}**`,
    "",
    `“${picked}” matches multiple teachers. Please select one of the following:`,
    "",
    list,
    "",
    "Reply with the full name to continue."
  ].join("\n");
}
