export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, OPTIONS",
          "Access-Control-Allow-Headers": "*",
        },
      });
    }

    try {
      const { searchParams } = new URL(request.url);
      const videoUrl = searchParams.get("url");

      if (!videoUrl || !/^https:\/\/(www\.)?youtube\.com\/watch/.test(videoUrl)) {
        return respond({ error: "Invalid or missing YouTube URL." }, 400);
      }

      const html = await fetchHTML(videoUrl);
      const player = extractPlayerResponse(html);
      
      if (!player) {
        console.error("Failed to extract player response");
        throw new Error("No ytInitialPlayerResponse found. YouTube may have changed their page structure.");
      }

      if (!player.streamingData) {
        throw new Error("No streaming data available. Video may be unavailable, private, or region-locked.");
      }

      const baseJsUrl = extractBaseJsUrl(html);
      const rawFormats = [
        ...(player.streamingData?.formats || []),
        ...(player.streamingData?.adaptiveFormats || []),
      ];

      if (rawFormats.length === 0) {
        throw new Error("No formats found in streaming data.");
      }

      let decipher = null;
      if (baseJsUrl) {
        try {
          decipher = await buildDecipher(baseJsUrl);
        } catch (e) {
          console.error("Decipher build failed:", e.message);
          // Continue without decipher - some videos may not need it
        }
      }

      const formats = await Promise.all(
        rawFormats.map(async (f) => {
          let url = f.url;
          if (!url && f.signatureCipher) {
            const params = new URLSearchParams(f.signatureCipher);
            url = params.get("url");
            const s = params.get("s");
            const sp = params.get("sp") || "sig";
            if (decipher && s) {
              const sig = decipher(s);
              url += `&${sp}=${sig}`;
            }
          }
          return {
            mime: f.mimeType?.split(";")[0],
            quality: f.qualityLabel || f.audioQuality || "unknown",
            ext: f.mimeType?.split("/")[1]?.split(";")[0] || "?",
            size: f.contentLength
              ? `${(f.contentLength / 1048576).toFixed(2)} MB`
              : "—",
            url,
          };
        })
      );

      const out = {
        title: player.videoDetails?.title || "Unknown",
        author: player.videoDetails?.author || "Unknown",
        duration: player.videoDetails?.lengthSeconds
          ? `${Math.floor(player.videoDetails.lengthSeconds / 60)}:${(
              "0" + (player.videoDetails.lengthSeconds % 60)
            ).slice(-2)}`
          : "—",
        formats: formats.filter((x) => x.url),
      };

      return respond(out);
    } catch (err) {
      console.error("Worker error:", err.message, err.stack);
      return respond({ 
        error: err.message,
        details: err.stack?.split('\n')[0] || "Unknown error"
      }, 500);
    }
  },
};

/* ---------------- Helper Functions ---------------- */

async function fetchHTML(url) {
  const res = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
    },
  });
  return res.text();
}

function extractPlayerResponse(html) {
  // Try multiple patterns for ytInitialPlayerResponse
  let m = html.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;/);
  if (!m) m = html.match(/var ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;/);
  if (!m) m = html.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*<\/script>/);
  
  if (!m) return null;
  
  try {
    // Find the matching closing brace
    let jsonStr = m[1];
    let depth = 0;
    let endIndex = 0;
    
    for (let i = 0; i < jsonStr.length; i++) {
      if (jsonStr[i] === '{') depth++;
      if (jsonStr[i] === '}') depth--;
      if (depth === 0) {
        endIndex = i + 1;
        break;
      }
    }
    
    jsonStr = jsonStr.substring(0, endIndex);
    return JSON.parse(jsonStr);
  } catch (e) {
    console.error("Failed to parse player response:", e);
    return null;
  }
}

function extractBaseJsUrl(html) {
  const m =
    html.match(/"jsUrl":"(\/s\/player\/[\w\d\/\-_\.]+base\.js)"/) ||
    html.match(/"js":"(\/s\/player\/[\w\d\/\-_\.]+base\.js)"/);
  return m ? "https://www.youtube.com" + m[1] : null;
}

/* ---------------- Cipher Decoding (v3) ---------------- */
async function buildDecipher(baseJsUrl) {
  const js = await (await fetch(baseJsUrl)).text();

  // Find main signature function name (supports 2023–2025 patterns)
  const funcNameMatch =
    js.match(/["']signature["']\s*,\s*([a-zA-Z0-9$]+)\(/) ||
    js.match(/\.sig\|\|([a-zA-Z0-9$]+)\(/) ||
    js.match(/["']s"\s*,\s*([a-zA-Z0-9$]{2,})\(/);
  const funcName = funcNameMatch ? funcNameMatch[1] : null;
  if (!funcName) throw new Error("Cipher function not found.");

  // Extract its body
  const bodyMatch = js.match(new RegExp(`${funcName}=function\\(a\\)\\{([^}]+)\\}`));
  if (!bodyMatch) throw new Error("Cipher body not found.");
  const body = bodyMatch[1];

  // Find helper object name and definition
  const helperNameMatch = body.match(/;([A-Za-z0-9$]{2})\./);
  const helperName = helperNameMatch ? helperNameMatch[1] : null;
  const helperDefMatch = js.match(
    new RegExp(`var ${helperName}=\\{(.*?)\\};`, "s")
  );
  if (!helperDefMatch) throw new Error("Helper not found.");
  const helperBody = helperDefMatch[1];

  const actions = {};
  helperBody.split("},").forEach((p) => {
    const m = p.match(/(\w+):function\(\w+(?:,\w+)?\)\{([^}]*)\}/);
    if (m) actions[m[1]] = m[2];
  });

  const ops = {};
  for (const [k, v] of Object.entries(actions)) {
    if (/splice/.test(v)) ops[k] = (a, b) => a.splice(0, b);
    else if (/reverse/.test(v)) ops[k] = (a) => a.reverse();
    else if (/var c=a\[0\];a\[0\]=a\[b%a\.length\];a\[b\]=c/.test(v))
      ops[k] = (a, b) => {
        const c = a[0];
        a[0] = a[b % a.length];
        a[b] = c;
      };
  }

  const steps = body.split(";").filter((s) => s.includes(helperName + "."));
  return function (sig) {
    const arr = sig.split("");
    for (const step of steps) {
      const m = step.match(/\.([a-zA-Z0-9$]+)\(a,?(\d+)?\)/);
      if (m) {
        const fn = ops[m[1]];
        const arg = parseInt(m[2]);
        fn && fn(arr, arg);
      }
    }
    return arr.join("");
  };
}

/* ---------------- Response ---------------- */
function respond(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "*",
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
