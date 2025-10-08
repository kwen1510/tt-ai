import express from "express";
import dotenv from "dotenv";
import axios from "axios";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import Groq from "groq-sdk";

dotenv.config();

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));

const { PORT, PASSWORD, APPS_SCRIPT_EXEC, GROQ_API_KEY } = process.env;

// --- Groq SDK client ---
const groq = new Groq({ apiKey: GROQ_API_KEY });

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
    const { question, results } = req.body;
    const collapsed = Array.isArray(results) ? JSON.stringify(results.slice(0, 5)) : JSON.stringify(results);

    const stream = await groq.chat.completions.create({
      model: "openai/gpt-oss-20b",
      messages: [
        { role: "system", content: "You are a helpful assistant for timetable queries." },
        { role: "user", content: `Answer ONLY from this JSON.\nQuestion: ${question}\nJSON (collapsed): ${collapsed}` }
      ],
      temperature: 0.2,
      top_p: 1,
      max_completion_tokens: 1024,
      stream: true
    });

    let full = "";
    for await (const chunk of stream) full += chunk.choices?.[0]?.delta?.content || "";
    res.json({ choices: [{ message: { content: full } }] });
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
      response_format: "verbose_json"
      // language: "en" // uncomment if you want to force language
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
