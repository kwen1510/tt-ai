import express from "express";
import dotenv from "dotenv";
import axios from "axios";
import multer from "multer";
import cors from "cors";
import FormData from "form-data";
import fs from "fs";
import Groq from "groq-sdk";

dotenv.config();

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public")); // serves /public/index.html

const {
  PORT,
  PASSWORD,
  APPS_SCRIPT_EXEC,
  GROQ_API_KEY,   // preferred for SDK
  GROQ_KEY        // fallback
} = process.env;

// ===== Groq SDK client (uses SDK's pattern) =====
const groq = new Groq({ apiKey: GROQ_API_KEY || GROQ_KEY });

// 1) Proxy user query to your Google Apps Script endpoint (backend holds password/keys)
app.post("/proxy", async (req, res) => {
  try {
    const { question, mode = "auto" } = req.body;
    if (!question) return res.status(400).json({ ok: false, error: "Missing question" });
    if (!APPS_SCRIPT_EXEC) return res.status(500).json({ ok: false, error: "Missing APPS_SCRIPT_EXEC" });

    const payload = { password: PASSWORD, mode, question };

    const r = await axios.post(APPS_SCRIPT_EXEC, payload, {
      headers: { "Content-Type": "application/json" },
    });

    return res.json(r.data);
  } catch (err) {
    console.error("Proxy error:", err?.response?.data || err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// 2) Summarize / answer directly from JSON using Groq SDK (stream server-side, return OpenAI-like shape)
app.post("/summarize", async (req, res) => {
  try {
    const { question, results } = req.body;
    const collapsed = Array.isArray(results)
      ? JSON.stringify(results.slice(0, 5)) // keep it small for prompt hygiene
      : JSON.stringify(results);

    const prompt = `
You answer timetable questions directly from JSON data.
Question: ${question}
JSON (collapsed): ${collapsed}
Respond clearly and concisely for a teacher/admin. If data is empty, say so.
    `.trim();

    // Groq SDK streaming (per docs)
    const stream = await groq.chat.completions.create({
      model: "openai/gpt-oss-20b",
      messages: [
        { role: "system", content: "You are a helpful assistant for timetable queries." },
        { role: "user", content: prompt }
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

    // Return in a structure your UI already reads: j?.choices?.[0]?.message?.content
    return res.json({ choices: [{ message: { content: full } }] });
  } catch (e) {
    console.error("Summarize error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// 3) Whisper Turbo transcription (Groq Audio API, OpenAI-compatible route)
app.post("/whisper", upload.single("audio"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });
  try {
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path));
    form.append("model", "whisper-large-v3-turbo");
    form.append("temperature", "0");

    const r = await axios.post("https://api.groq.com/openai/v1/audio/transcriptions", form, {
      headers: {
        ...form.getHeaders(),
        Authorization: `Bearer ${GROQ_API_KEY || GROQ_KEY}`,
      },
    });

    fs.unlinkSync(req.file.path);
    res.json(r.data);
  } catch (e) {
    console.error("Whisper error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// 4) Start server
const port = PORT || 8080;
app.listen(port, () => console.log(`✅ Server running on :${port}`));
