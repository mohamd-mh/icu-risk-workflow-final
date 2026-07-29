# Legacy / Archived Files

These files are **not part of the active application**. They are kept for
reference only and are not imported or rendered by `app.py`.

## legacy/agents/

An earlier, unused "Milestone 3" simulation of a multi-agent workflow:
`planner_agent.py`, `data_validator_agent.py`, `risk_assessor_agent.py`,
`verifier_agent.py`, `explainer_agent.py`, `trace_agent.py`.

Confirmed via `grep` before archiving: these modules only imported each
other (through `trace_agent.py`) and were never imported by `app.py`. The
live, active workflow used by the running application is
`agents/multi_agent_workflow.py` (see its module docstring).

## legacy/templates/

Template files never rendered by any Flask route in `app.py` (confirmed via
`grep -c 'render_template("<name>.html"' app.py` returning 0 for each):
`home.html`, `articles.html`, `dataset.html`, `architecture.html`,
`baseline.html`, `multi_agent.html`, `research_question.html`,
`stakeholders.html`, `ai_methodology.html`.

These are leftovers from an earlier documentation-site version of the app.
The routes that used to serve some of these paths now issue a redirect to
the current home page instead (see `legacy_documentation_redirect()` in
`app.py`), so no working URL is broken by moving these files here.

## What was intentionally NOT archived

The following routes/templates are not linked from the current sidebar
navigation (`NAV_SECTIONS` in `app.py`), but they are still actively wired
to a live Flask route and render successfully, so they were left in place:
`/demo` (`templates/demo.html`), `/team` (`templates/team.html`),
`/technologies` (`templates/technologies.html`),
`/testing-validation` (`templates/testing_validation.html`),
`/software-system` (`templates/software_system.html`), and
`/icu-dashboard` (`templates/icu_dashboard.html`). Moving or deleting these
would break a currently-working route, so they were left untouched.
