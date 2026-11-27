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
| NumPy, SciPy, librosa, soundfile | BSD / ISC |
| PyAV (כולל ספריות FFmpeg) | BSD (PyAV), LGPL (FFmpeg) |
| num2words, soxr | LGPL-2.1 (נטענות כספריות נפרדות) |
| python-docx, python-pptx, pypdf | MIT / BSD |

**הושמטו בכוונה:** ל־BlueTTS יש תלות ב־`phonemizer` וב־`espeak-ng`, ששתיהן ברישיון GPL. הן משמשות רק לשפות שאינן עברית, ולכן הן לא מותקנות ולא נכללות בקובצי ההתקנה.

## שכפול קול

השתמשו בשכפול קול רק עם הסכמה של בעל הקול. זו לא שאלה של זכויות יוצרים, אלא של זכות האדם על קולו. התוכנה מבקשת אישור לפני שמירה של קול משוכפל.
