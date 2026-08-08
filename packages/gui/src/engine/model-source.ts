/**
 * Where the model comes from, in one place.
 *
 * Two different answers are needed and they used to drift apart in two files:
 * the dev server serves the model itself (the range server in vite.config.ts),
 * while a static build has no such route and must fetch it from a host that
 * answers HTTP range requests with permissive CORS. HuggingFace does — verified
 * on a 709 MB LFS file: `206 Partial Content`, `accept-ranges: bytes`,
 * `access-control-allow-origin: *`, including after the CDN redirect. GitHub
 * Pages does not, which is why the model is not published with the site.
 *
 * Both are overridable at build time so a fork does not have to patch source.
 */

const MODEL_FILE = "LFM2.5-1.2B-Instruct-WQ4.wq4";
const HF_REPO = "karolrybak/LFM2.5-1.2B-Instruct-WQ4";

/** Human destination for "where do I get this file". */
export const MODEL_DOWNLOAD_URL = import.meta.env.VITE_MODEL_DOWNLOAD_URL
  ?? `https://huggingface.co/${HF_REPO}`;

/** Machine destination, read in ranges by HttpRangeSource. */
export const DEFAULT_MODEL_URL = import.meta.env.VITE_MODEL_URL
  ?? (import.meta.env.DEV
    // The dev server has the file locally; no reason to pull it over the wire.
    ? `${import.meta.env.BASE_URL}models/${MODEL_FILE}`
    : `https://huggingface.co/${HF_REPO}/resolve/main/${MODEL_FILE}`);

/** True when loading the default would mean a large cross-origin download. */
export const DEFAULT_IS_REMOTE = !import.meta.env.DEV;
