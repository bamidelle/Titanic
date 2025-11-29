:root{
  --font-sans: "Google Sans", "Roboto", "Inter", system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --color-primary: #1a73e8; /* Google-like blue for primary buttons */
  --surface-bg: #ffffff;
  --muted: #5f6368;
}

/* Base text */
body {
  font-family: var(--font-sans);
  font-size: 14px;        /* base body (paragraphs, secondary text) */
  line-height: 1.4;
  color: #202124;
}

/* Dashboard header / nav */
.app-header {
  font-size: 18px;        /* top-level nav / dashboard title */
  font-weight: 500;
  letter-spacing: 0.2px;
}

/* Card / list titles */
.card-title {
  font-size: 16px;        /* card titles or important list items */
  font-weight: 500;
}

/* Secondary or caption text */
.caption {
  font-size: 12px;
  color: var(--muted);
}

/* Buttons (primary & secondary) */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 36px;           /* typical small-to-medium button height */
  padding: 0 14px;        /* horizontal padding */
  border-radius: 4px;     /* subtle rounding like Google UIs */
  border: none;
  font-size: 14px;        /* Material default for buttons */
  font-weight: 500;
  text-transform: uppercase; /* Material often uses uppercase for buttons */
  cursor: pointer;
  line-height: 1;
  box-shadow: 0 1px 0 rgba(0,0,0,0.04); /* subtle elevation */
}

/* Primary button */
.btn-primary {
  background: var(--color-primary);
  color: white;
  min-width: 96px;
}

/* Secondary / ghost */
.btn-outline {
  background: transparent;
  color: var(--color-primary);
  border: 1px solid rgba(26,115,232,0.12);
}

/* Small / dense variant (e.g., toolbar icons) */
.btn-sm {
  height: 32px;
  padding: 0 10px;
  font-size: 13px;
  border-radius: 4px;
}
