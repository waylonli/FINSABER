# Documentation Deployment

The documentation site uses MkDocs Material.

## Install Docs Dependencies

```bash
pip install -e ".[docs]"
```

## Serve Locally

```bash
mkdocs serve
```

Open:

```text
http://127.0.0.1:8000
```

## Build Static Site

```bash
mkdocs build
```

The generated `site/` directory is ignored and should not be committed.

## Deploy To GitHub Pages

GitHub Pages deployment is handled by `.github/workflows/docs.yml`.

1. In GitHub, open repository **Settings > Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push changes to `main`, or run the `Deploy documentation` workflow manually.

After changing the Pages source, re-run the workflow from the GitHub **Actions** tab if a previous deployment attempt returned `404`.

The workflow installs `.[docs]`, runs `mkdocs build --strict`, uploads the generated `site/` artifact, and publishes it to Pages.

The expected public URL is:

```text
https://waylonli.github.io/FINSABER-2/
```

If you prefer manual branch deployment, run:

```bash
mkdocs gh-deploy
```

That fallback requires local GitHub credentials and publishes to `gh-pages`.
