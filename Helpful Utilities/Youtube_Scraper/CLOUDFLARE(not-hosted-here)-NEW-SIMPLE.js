// SIMPLE WORKER - Uses YouTube's Innertube API (same as yt-dlp)
// This is much more reliable than trying to parse HTML and decrypt signatures

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
        
        // Fetch the video stream from YouTube
        const videoResponse = await fetch(streamUrl, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
          },
        });
        
        console.log("Proxy: YouTube response status:", videoResponse.status);
        
        if (!videoResponse.ok) {
          return respond({ error: `YouTube returned ${videoResponse.status}` }, 500);
        }
        
        // Stream the video back
        return new Response(videoResponse.body, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": videoResponse.headers.get("Content-Type") || "video/mp4",
            "Content-Length": videoResponse.headers.get("Content-Length") || "",
            "Content-Disposition": `attachment; filename="${searchParams.get("filename") || "video.mp4"}"`,
            "Cache-Control": "no-store",
          },
        });
      }
      
      // INFO ENDPOINT - Use YouTube's Innertube API
      const videoUrl = searchParams.get("url");
      if (!videoUrl) {
        return respond({ error: "Missing URL parameter" }, 400);
      }

      // Extract video ID
      const videoIdMatch = videoUrl.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
      if (!videoIdMatch) {
        return respond({ error: "Invalid YouTube URL" }, 400);
      }
      const videoId = videoIdMatch[1];

      console.log("Fetching info for video:", videoId);

      // Call YouTube's Innertube API (official internal API)
      const playerResponse = await fetch("https://www.youtube.com/youtubei/v1/player?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        body: JSON.stringify({
          context: {
            client: {
              clientName: "WEB",
              clientVersion: "2.20241031.01.00",
            },
          },
          videoId: videoId,
        }),
      });

      if (!playerResponse.ok) {
        throw new Error(`Innertube API returned ${playerResponse.status}`);
      }

      const data = await playerResponse.json();
      
      if (!data.streamingData) {
        console.error("No streaming data:", data.playabilityStatus);
        throw new Error(data.playabilityStatus?.reason || "Video unavailable");
      }

      // Extract formats with direct URLs (no signature cipher)
      const rawFormats = [
        ...(data.streamingData?.formats || []),
        ...(data.streamingData?.adaptiveFormats || []),
      ];

      const formats = rawFormats
        .filter(f => f.url) // Only formats with direct URLs
        .map(f => ({
          mime: f.mimeType?.split(";")[0],
          quality: f.qualityLabel || f.audioQuality || "unknown",
          ext: f.mimeType?.split("/")[1]?.split(";")[0] || "?",
          size: f.contentLength ? `${(f.contentLength / 1048576).toFixed(2)} MB` : "—",
          url: f.url,
        }));

      const result = {
        title: data.videoDetails?.title || "Unknown",
        author: data.videoDetails?.author || "Unknown",
        duration: data.videoDetails?.lengthSeconds
          ? `${Math.floor(data.videoDetails.lengthSeconds / 60)}:${("0" + (data.videoDetails.lengthSeconds % 60)).slice(-2)}`
          : "—",
        formats,
      };

      console.log(`✅ Found ${formats.length} formats`);
      return respond(result);
      
    } catch (err) {
      console.error("Worker error:", err.message);
      return respond({ error: err.message }, 500);
    }
  },
};

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
