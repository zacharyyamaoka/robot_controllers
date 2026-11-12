# robotcontrollers-website

The robot_controllers landing page and docs: one page per black box (leaderboard, Pareto plots, the rung ladder, the
recommendation), one per controller (where its spring and damper live, parameters, a same-thread example run at build
time, per-rung results and plots, cost, prior art, W&B links), the rung ladder and the metric catalogue.
Every page is generated from `robot_controllers.registry`, `metrics.names`, each controller's config dataclass and
`results/<box-slug>/results.json` (CONTRACT §5); nothing is written per controller. The chrome (`site/site.css`,
`site/site.js`, `build/highlight.py`) is robotmsgs.com's, byte for byte; domain styles are in `site/rc.css`.

```bash
pnpm --dir /home/bam/robot_controllers/website run build           # dist/ from results/ ("not run yet" without it)
pnpm --dir /home/bam/robot_controllers/website run build:fixture   # dist/ from the synthetic test fixture
pnpm --dir /home/bam/robot_controllers/website run test            # builds into a temp dir, checks pages, numbers, links
pnpm --dir /home/bam/robot_controllers/website run serve           # build, then http://127.0.0.1:4494
```

- `build/build.py --results-dir DIR --dist DIR --origin URL`: the sitemap is written only with `--origin` (no domain yet).
- `tests/fixtures/results/` holds synthetic results-1 documents (`generated.fixture = true`; pages print a banner).
  Regenerate with `env -u VIRTUAL_ENV uv run --frozen --project /home/bam/robot_controllers --extra test python /home/bam/robot_controllers/website/tests/fixtures/make_fixtures.py`.
- `wrangler.jsonc` is a static-assets Worker on workers.dev only, no custom domain. It is not deployed.
