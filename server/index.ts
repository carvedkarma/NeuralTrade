import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes, hydrateBackfillStateFromDb, initializeStrategyLearner, backfillState } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import { checkIncompleteBackfillJobs, backfillHistoricalData } from "./historical-data";
import { loadPaperState } from "./paper/config";
import { loadCandleTimestamps } from "./unified-learning-controller";

const app = express();
const httpServer = createServer(app);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);

app.use(express.urlencoded({ extended: false }));

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`serving on port ${port}`);
      
      loadPaperState().then(() => {
        return hydrateBackfillStateFromDb();
      }).then(() => {
        return loadCandleTimestamps();
      }).then(() => {
        return initializeStrategyLearner();
      }).then(() => {
        return checkIncompleteBackfillJobs();
      }).then(async (result) => {
        if (result.hasIncomplete && result.progressPct !== undefined && result.progressPct < 100) {
          log(`Found incomplete backfill job (${result.progressPct}% complete) - auto-resuming...`);
          
          backfillState.inProgress = true;
          backfillState.progress = result.progressPct;
          backfillState.message = `Resuming from ${result.progressPct}%...`;
          
          try {
            const resumeResult = await backfillHistoricalData(
              result.symbol || "BTCUSDT",
              result.timeframe || "15m",
              370,
              (progress, message) => {
                backfillState.progress = progress;
                backfillState.message = message;
                if (progress % 10 === 0) {
                  log(`Backfill resume progress: ${progress}% - ${message}`);
                }
              }
            );
            backfillState.inProgress = false;
            backfillState.progress = 100;
            backfillState.message = `Complete! ${resumeResult.totalCandles} candles stored.`;
            log(`Backfill auto-resume complete: ${resumeResult.newCandles} candles added`);
          } catch (err: any) {
            backfillState.inProgress = false;
            backfillState.message = `Error: ${err.message}`;
            console.error("Error auto-resuming backfill:", err);
          }
        }
      }).catch((err) => {
        console.error("Error checking incomplete backfill jobs:", err);
      });
    },
  );
})();
