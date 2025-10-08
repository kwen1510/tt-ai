import express from "express";
import dotenv from "dotenv";
import axios from "axios";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import Groq from "groq-sdk";

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
} = process.env;

// ---- Groq SDK client ----
const groq = new Groq({ apiKey: GROQ_API_KEY });

// 1) Proxy to Google Apps Script (backend holds password/keys)
app.post("/proxy", async (req, res) => {
  try {
    const { question, mode = "auto" } = req.body;
    if (!APPS_SCRIPT_EXEC) return res.status(500).json({ ok: false, error: "Missing APPS_SCRIPT_EXEC" });
    if (!question) return res.status(400).json({ ok: false, error: "Missing question" });

    const payload = { password: PASSWORD, mode, question };
    const r = await axios.post(APPS_SCRIPT_EXEC, payload, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(r.data);
  } catch (err) {
    console.error("Proxy error:", err?.response?.data || err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// 2) Summarize/answer from JSON via Groq SDK (stream -> aggregate -> return)
app.post("/summarize", async (req, res) => {
  try {
    const { question, results } = req.body;
    const collapsed = Array.isArray(results) ? JSON.stringify(results.slice(0, 5)) : JSON.stringify(results);

    const stream = await groq.chat.completions.create({
      model: "openai/gpt-oss-20b",
      messages: [
        { role: "system", content: "You are a helpful assistant for timetable queries." },
        { role: "user", content: `Answer from this JSON only.\nQuestion: ${question}\nJSON (collapsed): ${collapsed}` }
      ],
      temperature: 0.2,
      top_p: 1,
      max_completion_tokens: 1024,
      stream: true
    });

    let full = "";
    for await (const chunk of stream) {
      full += chunk.choices?.[0]?.delta?.content || "";
    }
    res.json({ choices: [{ message: { content: full } }] });
  } catch (e) {
    console.error("Summarize error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// 3) Whisper Turbo transcription via Groq SDK (official docs pattern)
app.post("/whisper", upload.single("audio"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });
  try {
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(req.file.path),          // <- uses temp file from multer
      model: "whisper-large-v3-turbo",
      response_format: "verbose_json"
    });
    // Cleanup temp file
    try { fs.unlinkSync(req.file.path); } catch {}
    // Return text + segments for debugging if needed
    res.json({ text: transcription.text, segments: transcription.segments || [] });
  } catch (e) {
    console.error("Whisper error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT || 8080, () => {
  console.log(`✅ Server running on :${PORT || 8080}`);
});
