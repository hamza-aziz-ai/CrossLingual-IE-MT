"""
capture_screenshots.py
========================

Automates driving the running Streamlit app through the full
Input -> Translation & Alignment -> IE Extraction -> JSON/CSV Download
workflow with Playwright, saving one numbered PNG screenshot per step to
`screenshots/`. This produces the "Application Screenshots: step-by-step
screenshots demonstrating the complete end-to-end workflow" deliverable
required by the assignment brief.

Usage
-----
    # Terminal 1
    streamlit run frontend/app.py

    # Terminal 2 (after the app has finished loading in a browser tab once,
    # so model downloads on first click don't happen mid-capture)
    python -m playwright install chromium   # one-time
    python tools/capture_screenshots.py

Why a separate script instead of screenshotting by hand
-----------------------------------------------------------
Playwright driving the real, running app guarantees every screenshot
reflects actual application behaviour (not a mockup), is reproducible for
every group member's machine, and can be re-run any time the UI changes
without a manual re-capture pass.

Captured screens
-----------------
1. `01_initial_state.png`      — empty input, default configuration.
2. `02_input_entered.png`      — example sentence typed into the text area.
3. `03_translation_view.png`   — Translation & Alignment tab after running the pipeline.
4. `04_alignment_heatmap.png`  — the word-alignment heatmap sub-view.
5. `05_ner_view.png`           — Named Entities tab with colour-coded highlights.
6. `06_relations_view.png`     — Relations tab with the entity-relation graph.
7. `07_events_view.png`        — Events & Time tab.
8. `08_export_view.png`        — Export tab, ready to download JSON/CSV.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
APP_URL = "http://localhost:8501"

EXAMPLE_TEXT = (
    "Tim Cook announced the new iPhone in California yesterday. "
    "Apple will launch the product next week."
)


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright && "
              "python -m playwright install chromium")
        sys.exit(1)

    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    def snap(filename: str):
        """
        Save a screenshot covering the full rendered page, not just the
        1000px launch viewport.

        `page.screenshot(full_page=True)` normally expands the capture to
        `document.body.scrollHeight`, but Streamlit renders its content
        inside `[data-testid="stMain"]`, an internally-scrolling `<section>`
        that is fixed at the viewport height (`document.body.scrollHeight`
        stays 0) -- so `full_page=True` silently clips to 1000px instead of
        erroring. Every screenshot below the fold (the alignment heatmap,
        the bottom of the translation card, etc.) came out cropped until
        this was caught. Fix: read `stMain`'s true `scrollHeight`, force it
        to lay out at full height (`overflow: visible`, no `max-height`) so
        nothing stays clipped, then grow the viewport to match before
        capturing. The style override is undone immediately after so it
        doesn't affect the next interaction.
        """
        content_height = page.evaluate(
            """() => {
                const el = document.querySelector('[data-testid="stMain"]');
                const h = el.scrollHeight;
                el.style.setProperty('height', h + 'px', 'important');
                el.style.setProperty('max-height', 'none', 'important');
                el.style.setProperty('overflow', 'visible', 'important');
                return h;
            }"""
        )
        page.set_viewport_size({"width": 1600, "height": content_height + 100})
        page.wait_for_timeout(200)
        page.screenshot(path=str(SCREENSHOTS_DIR / filename))
        page.evaluate(
            """() => {
                const el = document.querySelector('[data-testid="stMain"]');
                el.style.removeProperty('height');
                el.style.removeProperty('max-height');
                el.style.removeProperty('overflow');
            }"""
        )
        page.set_viewport_size({"width": 1600, "height": 1000})
        print(f"Saved {filename}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        print(f"Opening {APP_URL} ...")
        page.goto(APP_URL, timeout=60_000)
        page.wait_for_timeout(3000)
        snap("01_initial_state.png")

        # Type the example sentence into the text area.
        textarea = page.locator("textarea").first
        textarea.click()
        textarea.fill(EXAMPLE_TEXT)
        page.wait_for_timeout(500)
        snap("02_input_entered.png")

        # Click "Translate & Extract" and wait for the pipeline to finish.
        # First run can take several minutes (model downloads); subsequent
        # runs are much faster. The wait below polls for the results tabs
        # to appear rather than using a fixed sleep.
        run_button = page.get_by_role("button", name="Translate & Extract")
        run_button.click()
        print("Pipeline running (this can take a while on first run -- "
              "downloading model weights)...")

        page.wait_for_selector("text=Translation & Alignment", timeout=600_000)
        page.wait_for_timeout(1500)
        snap("03_translation_view.png")

        # Expand the alignment heatmap view if a "Heatmap" sub-tab is present.
        try:
            page.get_by_text("Heatmap", exact=True).first.click(timeout=5000)
            page.wait_for_timeout(1000)
            snap("04_alignment_heatmap.png")
        except Exception:
            print("Heatmap sub-tab not found/clickable; skipping 04_alignment_heatmap.png")

        for tab_name, filename in [
            ("Named Entities", "05_ner_view.png"),
            ("Relations", "06_relations_view.png"),
            ("Events & Time", "07_events_view.png"),
            ("Export", "08_export_view.png"),
        ]:
            try:
                page.get_by_role("tab", name=tab_name).click(timeout=5000)
                page.wait_for_timeout(1200)
                snap(filename)
            except Exception as exc:
                print(f"Could not capture tab '{tab_name}': {exc}")

        browser.close()

    print(f"\nDone. Screenshots saved to: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
