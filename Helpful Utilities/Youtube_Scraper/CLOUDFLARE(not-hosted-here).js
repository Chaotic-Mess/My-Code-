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
        const videoId = searchParams.get("vid") || "";
        const desiredQuality = searchParams.get("quality") || "";
        const desiredExt = (searchParams.get("ext") || "").toLowerCase();
        if (!streamUrl) {
          return respond({ error: "Missing download URL" }, 400);
        }
        
        console.log("Proxy: Fetching video from:", streamUrl.substring(0, 100) + "...");
        
        const makeResponse = async (videoResponse) => new Response(videoResponse.body, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": videoResponse.headers.get("Content-Type") || "video/mp4",
            "Content-Length": videoResponse.headers.get("Content-Length") || "",
            "Content-Disposition": `attachment; filename="${searchParams.get("filename") || "video.mp4"}"`,
            "Cache-Control": "no-store",
          },
        });

        try {
          // Prefer a range request (some googlevideo endpoints require it)
          const headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com/",
            "Origin": "https://www.youtube.com",
            "Range": "bytes=0-" 
          };

          let videoResponse = await fetch(streamUrl, { headers });
          console.log("Proxy: YouTube response status:", videoResponse.status);
          console.log("Proxy: Content-Type:", videoResponse.headers.get("Content-Type"));
          console.log("Proxy: Content-Length:", videoResponse.headers.get("Content-Length"));

          // Retry without Range if first attempt failed
          if (!videoResponse.ok) {
            console.log("Proxy: retry without Range header");
            const { Range, ...h2 } = headers;
            videoResponse = await fetch(streamUrl, { headers: h2 });
            console.log("Proxy: retry status:", videoResponse.status);
          }

          if (videoResponse.ok) {
            return await makeResponse(videoResponse);
          }

          // Final fallback: if videoId provided, try Piped proxy stream
          if (videoId) {
            console.log("Proxy: direct fetch failed; attempting Piped fallback for", videoId);
            const piped = await fetchPipedStreams(videoId);
            // Find a best match by ext/quality; else first available
            let candidate = null;
            if (piped && piped.length) {
              candidate = piped.find(p => desiredExt && p.ext && p.ext.toLowerCase() === desiredExt) ||
                          piped.find(p => desiredQuality && (p.quality||"").includes(desiredQuality)) ||
                          piped[0];
            }
            if (candidate && candidate.url) {
              console.log("Proxy: fetching Piped candidate", candidate.url.substring(0, 80));
              const resp2 = await fetch(candidate.url, {
                headers: {
                  "User-Agent": headers["User-Agent"],
                  "Accept": "*/*",
                }
              });
              if (resp2.ok) return await makeResponse(resp2);
              console.log("Proxy: Piped candidate failed with", resp2.status);
            }
          }

          const errorText = await videoResponse.text().catch(() => "");
          console.error("Proxy: YouTube error:", (errorText||"").substring(0, 200));
          return respond({ 
            error: `YouTube returned ${videoResponse.status}`
          }, 500);
        } catch (err) {
          console.error("Proxy: Fetch error:", err.message);
          return respond({ error: "Proxy fetch failed: " + err.message }, 500);
        }
      }
      
      // Original info endpoint (reworked): Prefer Innertube API, then Piped, finally HTML scrape
      const videoUrl = searchParams.get("url");
      if (!videoUrl || !/^https:\/\/(www\.)?youtube\.com\/watch/.test(videoUrl)) {
        return respond({ error: "Invalid or missing YouTube URL." }, 400);
      }
      const videoIdMatch = videoUrl.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
      const videoId = videoIdMatch ? videoIdMatch[1] : null;
      
      let finalFormats = [];
      let metaTitle = "Unknown";
      let metaAuthor = "Unknown";
      let metaLength = null;

      // 1) Try Innertube API (no cipher needed for direct URLs)
      try {
        if (videoId) {
          const it = await fetchInnertubeData(videoId);
          if (it && it.streamingData) {
            const raw = [
              ...(it.streamingData.formats || []),
              ...(it.streamingData.adaptiveFormats || []),
            ];
            finalFormats = raw.filter(f => f.url).map(f => ({
              mime: f.mimeType?.split(";")[0],
              quality: f.qualityLabel || f.audioQuality || "unknown",
              ext: f.mimeType?.split("/")[1]?.split(";")[0] || "?",
              size: f.contentLength ? `${(f.contentLength / 1048576).toFixed(2)} MB` : "—",
              url: f.url,
            }));
            metaTitle = it.videoDetails?.title || metaTitle;
            metaAuthor = it.videoDetails?.author || metaAuthor;
            metaLength = it.videoDetails?.lengthSeconds || metaLength;
          }
        }
      } catch (e) {
        console.log("Innertube fetch failed:", e.message);
      }

      // 2) If still nothing, try HTML scrape (best-effort; may not decipher)
      if (!finalFormats.length) {
        try {
          const html = await fetchHTML(videoUrl);
          const player = extractPlayerResponse(html);
          if (player?.streamingData) {
            const baseJsUrl = extractBaseJsUrl(html);
            let decipher = null;
            if (baseJsUrl) {
              try {
                decipher = await buildDecipher(baseJsUrl);
              } catch (e) {
                console.log("Decipher build failed:", e.message);
              }
            }
            const raw = [
              ...(player.streamingData.formats || []),
              ...(player.streamingData.adaptiveFormats || []),
            ];
            const mapped = await Promise.all(raw.map(async (f) => {
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
                size: f.contentLength ? `${(f.contentLength / 1048576).toFixed(2)} MB` : "—",
                url,
              };
            }));
            finalFormats = mapped.filter(x => x.url);
            metaTitle = player.videoDetails?.title || metaTitle;
            metaAuthor = player.videoDetails?.author || metaAuthor;
            metaLength = player.videoDetails?.lengthSeconds || metaLength;
          }
        } catch (e) {
          console.log("HTML scrape failed:", e.message);
        }
      }

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

      // 3) Piped fallback if still nothing
      if (!finalFormats.length && videoId) {
        try {
          console.log("No direct formats; trying Piped for", videoId);
          const piped = await fetchPipedStreams(videoId);
          if (piped && piped.length) {
            finalFormats = piped;
            console.log(`Piped fallback provided ${piped.length} formats`);
          }
        } catch (e) {
          console.log("Piped fallback failed:", e.message);
        }
      }

      const out = {
        title: metaTitle,
        author: metaAuthor,
        duration: metaLength
          ? `${Math.floor(metaLength / 60)}:${("0" + (metaLength % 60)).slice(-2)}`
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
    console.log("Tried all patterns, none matched");
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

/* ---------------- Innertube (YouTube internal) ---------------- */
async function fetchInnertubeData(videoId) {
  const key = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"; // public web API key used by YouTube WEB client
  const url = `https://www.youtube.com/youtubei/v1/player?key=${key}`;
  const body = {
    context: {
      client: {
        clientName: "WEB",
        clientVersion: "2.20241031.01.00",
      },
    },
    videoId,
  };
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Innertube ${res.status}`);
  return res.json();
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
