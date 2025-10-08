import express from "express";
import dotenv from "dotenv";
import axios from "axios";
import multer from "multer";
import cors from "cors";
import FormData from "form-data";
import fs from "fs";

dotenv.config();

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public")); // serves index.html

const { PORT, PASSWORD, APPS_SCRIPT_EXEC, GROQ_KEY } = process.env;

// 1️⃣ Proxy user query to your Google Apps Script endpoint
app.post("/proxy", async (req, res) => {
  try {
    const { question, mode = "auto" } = req.body;
    if (!question) return res.status(400).json({ ok: false, error: "Missing question" });

    const payload = { password: PASSWORD, mode, question };
    const r = await axios.post(APPS_SCRIPT_EXEC, payload, {
      headers: { "Content-Type": "application/json" },
    });
    return res.json(r.data);
  } catch (err) {
    console.error("Proxy error:", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// 2️⃣ Summarize or answer directly using Groq LLM (openai/gpt-oss-20b)
app.post("/summarize", async (req, res) => {
  try {
    const { question, results } = req.body;
    const collapsed = Array.isArray(results)
      ? JSON.stringify(results.slice(0, 5))
      : JSON.stringify(results);

    const prompt = `
You are an assistant that answers timetable questions directly from JSON data.
Question: ${question}
JSON data (collapsed): ${collapsed}
Answer clearly, concisely, and in plain English.
`;

    const r = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "openai/gpt-oss-20b",
        temperature: 0.2,
        messages: [
          { role: "system", content: "You are a helpful assistant for timetable queries." },
          { role: "user", content: prompt },
        ],
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${GROQ_KEY}`,
        },
      }
    );
    res.json(r.data);
  } catch (e) {
    console.error("Summarize error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// 3️⃣ Whisper Turbo transcription (Groq)
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
        Authorization: `Bearer ${GROQ_KEY}`,
      },
    });
    fs.unlinkSync(req.file.path);
    res.json(r.data);
  } catch (e) {
    console.error("Whisper error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// 4️⃣ Start the server
const port = PORT || 8080;
app.listen(port, () => console.log(`✅ Server running on port ${port}`));
