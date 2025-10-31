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
      const { searchParams, pathname } = new URL(request.url);
      
      // DOWNLOAD PROXY ENDPOINT
      if (pathname === "/download" || searchParams.has("download")) {
        const streamUrl = searchParams.get("download") || searchParams.get("url");
        if (!streamUrl) {
          return respond({ error: "Missing download URL" }, 400);
        }
        
        console.log("Proxy: Fetching video from:", streamUrl.substring(0, 100) + "...");
        
        try {
          // Fetch the video stream from YouTube with proper headers
          const videoResponse = await fetch(streamUrl, {
            headers: {
              "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
              "Accept": "*/*",
              "Accept-Language": "en-US,en;q=0.9",
            },
          });
          
          console.log("Proxy: YouTube response status:", videoResponse.status);
          console.log("Proxy: Content-Type:", videoResponse.headers.get("Content-Type"));
          console.log("Proxy: Content-Length:", videoResponse.headers.get("Content-Length"));
          
          if (!videoResponse.ok) {
            const errorText = await videoResponse.text();
            console.error("Proxy: YouTube error:", errorText.substring(0, 200));
            return respond({ 
              error: `YouTube returned ${videoResponse.status}`,
              details: errorText.substring(0, 200)
            }, 500);
          }
          
          // Return the video stream with appropriate headers
          return new Response(videoResponse.body, {
            headers: {
              "Access-Control-Allow-Origin": "*",
              "Content-Type": videoResponse.headers.get("Content-Type") || "video/mp4",
              "Content-Length": videoResponse.headers.get("Content-Length") || "",
              "Content-Disposition": `attachment; filename="${searchParams.get("filename") || "video.mp4"}"`,
              "Cache-Control": "no-store",
            },
          });
        } catch (err) {
          console.error("Proxy: Fetch error:", err.message);
          return respond({ error: "Proxy fetch failed: " + err.message }, 500);
        }
      }
      
      // Original info endpoint
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

      let finalFormats = formats.filter((x) => x.url);

      // Fallback: if no direct formats found, try Piped API (yt-dlp style backend)
      if (!finalFormats.length) {
        try {
          const vid = (videoUrl.match(/[?&]v=([a-zA-Z0-9_-]{11})/) || [null, null])[1];
          if (vid) {
            console.log("No direct formats; trying Piped for", vid);
            const piped = await fetchPipedStreams(vid);
            if (piped && piped.length) {
              finalFormats = piped;
              console.log(`Piped fallback provided ${piped.length} formats`);
            } else {
              console.log("Piped fallback returned 0 formats");
            }
          }
        } catch (e) {
          console.error("Piped fallback failed:", e.message);
        }
      }

      const out = {
        title: player.videoDetails?.title || "Unknown",
        author: player.videoDetails?.author || "Unknown",
        duration: player.videoDetails?.lengthSeconds
          ? `${Math.floor(player.videoDetails.lengthSeconds / 60)}:${(
              "0" + (player.videoDetails.lengthSeconds % 60)
            ).slice(-2)}`
          : "—",
        formats: finalFormats,
      };

      console.log(`Found ${out.formats.length} formats`);
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

/* ---------------- Cipher Decoding (v4 - Enhanced) ---------------- */
async function buildDecipher(baseJsUrl) {
  const js = await (await fetch(baseJsUrl)).text();

  // Try multiple patterns to find the signature function (YouTube keeps changing this)
  const patterns = [
    /\b[cs]\s*&&\s*[adf]\.set\([^,]+\s*,\s*encodeURIComponent\(([a-zA-Z0-9$]+)\(/,
    /\b[a-zA-Z0-9]+\s*&&\s*[a-zA-Z0-9]+\.set\([^,]+\s*,\s*encodeURIComponent\(([a-zA-Z0-9$]+)\(/,
    /(?:\b|[^a-zA-Z0-9$])([a-zA-Z0-9$]{2,})\s*=\s*function\(\s*a\s*\)\s*\{\s*a\s*=\s*a\.split\(\s*""\s*\)/,
    /([a-zA-Z0-9$]+)\s*=\s*function\(\s*a\s*\)\s*\{\s*a\s*=\s*a\.split\(\s*""\s*\);[a-zA-Z0-9$]{2}\.[a-zA-Z0-9$]{2}\(a,\d+\)/,
    /\b[cs]\s*&&\s*[adf]\.set\([^,]+\s*,\s*([a-zA-Z0-9$]+)\(/,
    /\bc\s*&&\s*a\.set\([^,]+\s*,\s*\([^)]*\)\s*\(\s*([a-zA-Z0-9$]+)\(/,
    /["']signature["']\s*,\s*([a-zA-Z0-9$]+)\(/,
    /\.sig\|\|([a-zA-Z0-9$]+)\(/,
    /["']s"\s*,\s*([a-zA-Z0-9$]{2,})\(/,
  ];

  let funcName = null;
  for (const pattern of patterns) {
    const match = js.match(pattern);
    if (match && match[1]) {
      funcName = match[1];
      console.log("Found cipher function:", funcName, "using pattern:", pattern.source.substring(0, 50));
      break;
    }
  }
  
  if (!funcName) {
    console.error("Tried all patterns, none matched");
    throw new Error("Cipher function not found.");
  }

  // Extract function body with multiple pattern attempts
  let bodyMatch = js.match(new RegExp(`${funcName.replace(/\$/g, '\\$')}=function\\(a\\)\\{([^}]+)\\}`));
  if (!bodyMatch) {
    bodyMatch = js.match(new RegExp(`${funcName.replace(/\$/g, '\\$')}\\s*=\\s*function\\s*\\(\\s*a\\s*\\)\\s*\\{(.*?)\\}`, 's'));
  }
  if (!bodyMatch) {
    console.error("Could not find function body for:", funcName);
    throw new Error("Cipher body not found.");
  }
  const body = bodyMatch[1];
  console.log("Function body:", body.substring(0, 100));

  // Find helper object name and definition
  const helperNameMatch = body.match(/;([A-Za-z0-9$]{2,3})\./);
  const helperName = helperNameMatch ? helperNameMatch[1] : null;
  if (!helperName) {
    console.error("Could not find helper object name in body");
    throw new Error("Helper object not found.");
  }
  
  console.log("Helper object name:", helperName);
  
  // Try multiple patterns for helper definition
  let helperDefMatch = js.match(new RegExp(`var ${helperName}=\\{(.*?)\\};`, "s"));
  if (!helperDefMatch) {
    helperDefMatch = js.match(new RegExp(`${helperName}=\\{(.*?)\\};`, "s"));
  }
  if (!helperDefMatch) {
    console.error("Could not find helper definition for:", helperName);
    throw new Error("Helper definition not found.");
  }
  const helperBody = helperDefMatch[1];

  const actions = {};
  helperBody.split("},").forEach((p) => {
    const m = p.match(/(\w+):function\(\w+(?:,\w+)?\)\{([^}]*)\}/);
    if (m) actions[m[1]] = m[2];
  });

  console.log("Found actions:", Object.keys(actions));

  const ops = {};
  for (const [k, v] of Object.entries(actions)) {
    if (/splice/.test(v)) {
      ops[k] = (a, b) => a.splice(0, b);
      console.log(`  ${k}: splice`);
    } else if (/reverse/.test(v)) {
      ops[k] = (a) => a.reverse();
      console.log(`  ${k}: reverse`);
    } else if (/var c=a\[0\];a\[0\]=a\[b%a\.length\];a\[b\]=c/.test(v) || /var c=a\[0\];a\[0\]=a\[b(?:%a\.length)?\];a\[b(?:%a\.length)?\]=c/.test(v)) {
      ops[k] = (a, b) => {
        const c = a[0];
        a[0] = a[b % a.length];
        a[b] = c;
      };
      console.log(`  ${k}: swap`);
    }
  }
  
  if (Object.keys(ops).length === 0) {
    console.error("No operations found! Actions:", actions);
    throw new Error("No cipher operations decoded");
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

/* ---------------- Piped Fallback ---------------- */
async function fetchPipedStreams(videoId) {
  const INSTANCES = [
    "https://piped.video",
    "https://pipedapi.kavin.rocks",
    "https://piped.privacydev.net",
  ];

  for (const host of INSTANCES) {
    try {
      const url = `${host}/api/v1/streams/${videoId}`;
      console.log("Piped: fetching", url);
      const res = await fetch(url, {
        headers: {
          "Accept": "application/json",
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
      });
      if (!res.ok) {
        console.log("Piped instance returned", res.status);
        continue;
      }
      const data = await res.json();

      const out = [];

      const pushStream = (s, kind) => {
        if (!s) return;
        const url = s.url || s.proxyUrl || s.downloadUrl;
        if (!url) return;
        const container = (s.container || s.codec || "").toLowerCase();
        const mime = s.mimeType || (kind === "audio" ? `audio/${container || "m4a"}` : `video/${container || "mp4"}`);
        const ext = (mime.split("/")[1] || container || "mp4").split(";")[0];
        const size = s.size || s.contentLength;
        const quality = s.qualityLabel || s.quality || s.bitrate || "unknown";
        out.push({
          mime,
          quality: String(quality),
          ext,
          size: size ? `${(Number(size) / 1048576).toFixed(2)} MB` : "—",
          url,
        });
      };

      (data.videoStreams || []).forEach(v => pushStream(v, "video"));
      (data.audioStreams || []).forEach(a => pushStream(a, "audio"));
      (data.relatedStreams || []); // ignore

      // Some instances provide combined formats
      if (Array.isArray(data.streams)) data.streams.forEach(v => pushStream(v, "video"));

      return out;
    } catch (e) {
      console.log("Piped fetch failed:", e.message);
      continue;
    }
  }

  return [];
}
