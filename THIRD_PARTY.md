# רכיבי צד שלישי ורישיונות

הקוד של קול־סטודיו הוא ברישיון MIT (ראו `LICENSE`). התוכנה משתמשת ברכיבים ובמודלים של אחרים, וכל אחד מהם נשאר תחת הרישיון שלו.

## מודלים (יורדים בהפעלה הראשונה, לא כלולים בקובץ ההתקנה)

| מודל | שימוש | רישיון | הערות |
|---|---|---|---|
| [BlueV3](https://huggingface.co/notmax123/BlueV3-onnx) · Max Melichov | הקראה ושכפול קול | MIT | רכיב השמע (vocoder) נגזר מ־Supertonic-3 וכפוף ל־**OpenRAIL-M**: מגבלות שימוש לרעה, ללא הגבלה מסחרית |
| קולות מוכנים של BlueV3 | 15 קולות | CC BY 4.0 (LibriTTS-R) | חובה לציין את המקור |
| [RenikudPlus](https://huggingface.co/notmax123/RenikudPlus) | הגייה (G2P) | Apache-2.0 | |
| מילון אנגלית של polyg2p | מילים באנגלית בתוך טקסט עברי | BSD (נגזר מ־CMU Pronouncing Dictionary) | |
| [Phonikud](https://huggingface.co/Phonikud/phonikud-onnx) | ניקוד | ראו את דף המודל | מבוסס על מודל הניקוד של [Dicta](https://huggingface.co/dicta-il) |
| [ivrit.ai Whisper](https://huggingface.co/ivrit-ai) | תמלול | ראו את דף המודל | מבוסס Whisper של OpenAI (MIT) |
| [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) + [התוספת העברית](https://huggingface.co/notmax123/Fish-Audio-S2-Pro-He) · מנוע אופציונלי | הקראה עם רגשות | **Fish Audio Research License** | מחקר ושימוש לא מסחרי בלבד. מותקן רק אחרי אישור המשתמש, ולא נכלל בקובצי ההתקנה. "This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved." **Built with Fish Audio.** |
| [Qwen3-TTS 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) + [QwenTTS-he](https://huggingface.co/notmax123/QwenTTS-he-1.7B) · מנוע אופציונלי | הקראה ושכפול קול | Apache-2.0 | מותקן רק לבקשת המשתמש |
| [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) · OpenMOSS · מנוע אופציונלי | אפקטים קוליים | Apache-2.0 | מותקן רק לבקשת המשתמש. התאמה ל־Apple GPU לפי הרעיון של [MOSS-SoundEffect_v2.0_MPS_ROCm](https://github.com/VladimirTalyzin/MOSS-SoundEffect_v2.0_MPS_ROCm) (Apache-2.0) |
| [Stable Audio 3 Small-SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) · Stability AI · מנוע אופציונלי | אפקטים קוליים | **Stability AI Community License** + תנאי Gemma (רכיב T5Gemma) | חינם למחקר, לשימוש לא מסחרי ולעסקים עם הכנסה שנתית של פחות ממיליון דולר. מעל זה צריך רישיון מ־Stability AI. המשתמש מאשר את התנאים בעצמו ב־Hugging Face. "Powered by Stability AI." |

## ספריות קוד

| ספרייה | רישיון |
|---|---|
| [BlueTTS / blue-onnx](https://github.com/maxmelichov/BlueTTS) | MIT |
| renikud-plus, phonikud-onnx, heb-tts-normalizer, hebrew-num2words | MIT / Apache-2.0 |
| faster-whisper, CTranslate2 | MIT |
| ONNX Runtime | MIT |
| FastAPI, Uvicorn, pywebview, psutil | MIT / BSD |
| MCP Python SDK (Model Context Protocol) | MIT |
| polyg2p (מילים באנגלית בתוך עברית; נכלל במק בלבד) | Apache-2.0 |
| קוד Fish Speech (fork של maxmelichov, מורד בהתקנת המנוע האופציונלי) | Fish Audio Research License |
| uv (מכין את סביבת המנוע האופציונלי) | MIT / Apache-2.0 |
| NumPy, SciPy, librosa, soundfile | BSD / ISC |
| PyAV (כולל ספריות FFmpeg) | BSD (PyAV), LGPL (FFmpeg) |
| num2words, soxr | LGPL-2.1 (נטענות כספריות נפרדות) |
| python-docx, python-pptx, pypdf | MIT / BSD |

**הושמטו בכוונה:** ל־BlueTTS יש תלות ב־`phonemizer` וב־`espeak-ng`, ששתיהן ברישיון GPL. הן משמשות רק לשפות שאינן עברית, ולכן הן לא מותקנות ולא נכללות בקובצי ההתקנה.

## שכפול קול

השתמשו בשכפול קול רק עם הסכמה של בעל הקול. זו לא שאלה של זכויות יוצרים, אלא של זכות האדם על קולו. התוכנה מבקשת אישור לפני שמירה של קול משוכפל.

## עיצוב

| רכיב | רישיון |
|---|---|
| גופן [Rubik](https://github.com/googlefonts/rubik) | SIL Open Font License 1.1 (`kol_studio/assets/fonts/OFL.txt`) |
| סמלים מ־[Lucide](https://lucide.dev) | ISC |
| איור המיקרופון והאוזניות | מתוך תמונת העיצוב של הפרויקט |
