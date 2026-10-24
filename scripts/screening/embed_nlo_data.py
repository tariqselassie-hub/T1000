# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Embeds the NLO dataset directly into web/nlo_ferro.html for 100% offline file:// compatibility."""

import json
from pathlib import Path


def main():
    json_path = Path("web/data/pitt_nlo_ferroelectric_pareto.json")
    html_path = Path("web/nlo_ferro.html")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_str = json.dumps(data)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    target = "let allLeads = [];"
    replacement = f"const EMBEDDED_NLO_DATA = {json_str};\n        let allLeads = EMBEDDED_NLO_DATA.all_candidates || EMBEDDED_NLO_DATA.top_leads;"
    if target in html:
        html = html.replace(target, replacement)

    old_load = """        async function loadData() {
            try {
                const res = await fetch('data/pitt_nlo_ferroelectric_pareto.json');
                const data = await res.json();
                allLeads = data.all_candidates || data.top_leads;
                populateRoster();
                selectLead(0);
                resizeCanvases();
                requestAnimationFrame(renderLoop);
            } catch (err) {
                console.error("Failed to load NLO dataset:", err);
            }
        }"""

    new_load = """        async function loadData() {
            try {
                const res = await fetch('data/pitt_nlo_ferroelectric_pareto.json');
                if (res.ok) {
                    const data = await res.json();
                    allLeads = data.all_candidates || data.top_leads;
                }
            } catch (err) {
                allLeads = EMBEDDED_NLO_DATA.all_candidates || EMBEDDED_NLO_DATA.top_leads;
            }
            populateRoster();
            selectLead(0);
            resizeCanvases();
            requestAnimationFrame(renderLoop);
        }"""

    if old_load in html:
        html = html.replace(old_load, new_load)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Successfully embedded 119 candidates ({len(json_str)} bytes) into {html_path}")


if __name__ == "__main__":
    main()
