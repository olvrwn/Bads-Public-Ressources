#!/usr/bin/env node

/**
 * Automated PR Review Agent
 *
 * Checks PRs tagged "automated" for:
 * 1. Only guides.json was modified
 * 2. Valid JSON structure (array of guide objects with exact required fields)
 * 3. No additional/missing properties per guide object
 * 4. All IDs are unique (used exactly once)
 *
 * On success: approves the PR and sends a green Slack message.
 * On failure: requests changes with a comment listing all issues,
 *             and sends a red Slack message.
 */

const https = require("https");

// ─── Config ──────────────────────────────────────────────────────────────────

const GITHUB_TOKEN  = process.env.GITHUB_TOKEN;
const SLACK_WEBHOOK = process.env.SLACK_WEBHOOK_URL;
const PR_NUMBER     = process.env.PR_NUMBER;
const [OWNER, REPO] = (process.env.REPO || "").split("/");

const REQUIRED_KEYS = new Set([
  "id",
  "title",
  "description",
  "language",
  "estimatedReadingMinutes",
  "source",
  "url",
  "author",
  "habits",
  "tags",
]);

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

function request(options, body = null) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", (c) => (data += c));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, body: data });
        }
      });
    });
    req.on("error", reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function github(method, path, body = null) {
  return request(
    {
      hostname: "api.github.com",
      path,
      method,
      headers: {
        Authorization: `Bearer ${GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "pr-review-agent",
        "Content-Type": "application/json",
      },
    },
    body
  );
}

function postSlack(payload) {
  const body = JSON.stringify(payload);
  const url = new URL(SLACK_WEBHOOK);
  return request(
    {
      hostname: url.hostname,
      path: url.pathname + url.search,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    },
    payload
  );
}

// ─── Validation ───────────────────────────────────────────────────────────────

function validateGuides(guides) {
  const issues = [];

  if (!Array.isArray(guides)) {
    issues.push("Root value must be a JSON **array** of guide objects.");
    return issues;
  }

  if (guides.length === 0) {
    issues.push("The guides array is empty — at least one entry is required.");
    return issues;
  }

  const seenIds = new Map(); // id → [indices]

  guides.forEach((entry, idx) => {
    const label = `Entry #${idx + 1} (id: "${entry?.id ?? "unknown"}")`;

    if (typeof entry !== "object" || Array.isArray(entry) || entry === null) {
      issues.push(`${label}: must be a plain object.`);
      return;
    }

    const entryKeys = new Set(Object.keys(entry));

    // Missing keys
    for (const key of REQUIRED_KEYS) {
      if (!entryKeys.has(key)) {
        issues.push(`${label}: missing required property \`${key}\`.`);
      }
    }

    // Extra keys
    for (const key of entryKeys) {
      if (!REQUIRED_KEYS.has(key)) {
        issues.push(`${label}: unexpected additional property \`${key}\`.`);
      }
    }

    // Type checks for present keys
    if (entryKeys.has("id") && typeof entry.id !== "string") {
      issues.push(`${label}: \`id\` must be a string.`);
    }
    if (entryKeys.has("title") && typeof entry.title !== "string") {
      issues.push(`${label}: \`title\` must be a string.`);
    }
    if (entryKeys.has("description") && typeof entry.description !== "string") {
      issues.push(`${label}: \`description\` must be a string.`);
    }
    if (entryKeys.has("language") && typeof entry.language !== "string") {
      issues.push(`${label}: \`language\` must be a string.`);
    }
    if (
      entryKeys.has("estimatedReadingMinutes") &&
      typeof entry.estimatedReadingMinutes !== "number"
    ) {
      issues.push(`${label}: \`estimatedReadingMinutes\` must be a number.`);
    }
    if (entryKeys.has("source") && typeof entry.source !== "string") {
      issues.push(`${label}: \`source\` must be a string.`);
    }
    if (entryKeys.has("url") && typeof entry.url !== "string") {
      issues.push(`${label}: \`url\` must be a string.`);
    }
    if (entryKeys.has("habits") && !Array.isArray(entry.habits)) {
      issues.push(`${label}: \`habits\` must be an array.`);
    }
    if (entryKeys.has("tags") && !Array.isArray(entry.tags)) {
      issues.push(`${label}: \`tags\` must be an array.`);
    }
    // author may be null or string
    if (
      entryKeys.has("author") &&
      entry.author !== null &&
      typeof entry.author !== "string"
    ) {
      issues.push(`${label}: \`author\` must be a string or null.`);
    }

    // Track ID for duplicate check
    if (entry.id !== undefined) {
      if (!seenIds.has(entry.id)) seenIds.set(entry.id, []);
      seenIds.get(entry.id).push(idx + 1);
    }
  });

  // Duplicate IDs
  for (const [id, indices] of seenIds) {
    if (indices.length > 1) {
      issues.push(
        `Duplicate ID \`"${id}"\` found at entries: ${indices.join(", ")}. Each ID must be unique.`
      );
    }
  }

  return issues;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`Reviewing PR #${PR_NUMBER} in ${OWNER}/${REPO} …`);

  // 1. Get changed files
  const filesRes = await github(
    "GET",
    `/repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/files`
  );
  const changedFiles = filesRes.body;

  const issues = [];

  // 2. Check only guides.json was modified
  const fileNames = changedFiles.map((f) => f.filename);
  console.log("Changed files:", fileNames);

  const nonGuides = fileNames.filter((f) => f !== "guides.json");
  if (nonGuides.length > 0) {
    issues.push(
      `Only \`guides.json\` should be modified. Unexpected files changed: ${nonGuides
        .map((f) => `\`${f}\``)
        .join(", ")}.`
    );
  }

  // 3. Validate guides.json content
  const guidesFile = changedFiles.find((f) => f.filename === "guides.json");
  if (!guidesFile) {
    issues.push(
      "`guides.json` was not modified in this PR — nothing to review."
    );
  } else {
    // Fetch the raw file content from the PR branch
    const rawRes = await github(
      "GET",
      `/repos/${OWNER}/${REPO}/contents/guides.json?ref=${encodeURIComponent(
        guidesFile.sha ? guidesFile.sha : "HEAD"
      )}`
    );

    let rawContent;
    try {
      rawContent = Buffer.from(rawRes.body.content, "base64").toString("utf-8");
    } catch {
      // Fallback: use blob sha directly
      const blobRes = await github(
        "GET",
        `/repos/${OWNER}/${REPO}/git/blobs/${guidesFile.sha}`
      );
      rawContent = Buffer.from(blobRes.body.content, "base64").toString("utf-8");
    }

    let parsed;
    try {
      parsed = JSON.parse(rawContent);
    } catch (e) {
      issues.push(`\`guides.json\` is **not valid JSON**: ${e.message}`);
      parsed = null;
    }

    if (parsed !== null) {
      const structureIssues = validateGuides(parsed);
      issues.push(...structureIssues);
    }
  }

  // 4. Post review to GitHub
  const approved = issues.length === 0;
  const reviewEvent = approved ? "APPROVE" : "REQUEST_CHANGES";

  let reviewBody;
  if (approved) {
    reviewBody =
      "✅ **Automated PR Review — Passed**\n\n" +
      "All checks passed:\n" +
      "- Only `guides.json` was modified\n" +
      "- Valid JSON structure\n" +
      "- All required properties present, no extra properties\n" +
      "- All IDs are unique\n\n" +
      "This PR is approved and ready to merge.";
  } else {
    const issueList = issues.map((i) => `- ${i}`).join("\n");
    reviewBody =
      "❌ **Automated PR Review — Issues Detected**\n\n" +
      "The following issues must be fixed before this PR can be merged:\n\n" +
      issueList;
  }

  const reviewRes = await github(
    "POST",
    `/repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/reviews`,
    { event: reviewEvent, body: reviewBody }
  );
  console.log(`GitHub review submitted: ${reviewRes.status} — ${reviewEvent}`);

  // 5. Send Slack notification
  const prUrl = `https://github.com/${OWNER}/${REPO}/pull/${PR_NUMBER}`;
  const slackColor = approved ? "#2eb886" : "#e01e5a";
  const slackTitle = approved
    ? "✅ Automated PR Approved"
    : "❌ Automated PR — Issues Detected";
  const slackText = approved
    ? `PR #${PR_NUMBER} passed all checks and has been approved.`
    : `PR #${PR_NUMBER} has ${issues.length} issue(s) that need to be fixed:\n` +
      issues.map((i) => `• ${i}`).join("\n");

  await postSlack({
    attachments: [
      {
        color: slackColor,
        blocks: [
          {
            type: "section",
            text: { type: "mrkdwn", text: `*${slackTitle}*` },
          },
          {
            type: "section",
            text: { type: "mrkdwn", text: slackText },
          },
          {
            type: "section",
            fields: [
              { type: "mrkdwn", text: `*Repository:*\n${OWNER}/${REPO}` },
              { type: "mrkdwn", text: `*PR:*\n<${prUrl}|#${PR_NUMBER}>` },
            ],
          },
        ],
      },
    ],
  });
  console.log("Slack notification sent.");

  // 6. Exit with error code if issues found (fails the CI step for visibility)
  if (!approved) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error in review agent:", err);
  process.exit(1);
});