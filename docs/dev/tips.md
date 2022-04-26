# Cheat sheet

**Add new Python package:**
Add the package to `gradient/requirements.txt`,
try if it works by running `pip install -r gradient/requirements.txt`,
and rebuild the [development Docker image](env.md).
Also freeze exact versions of all packages by running
`conda list --export > awe/requirements.txt`.

**Kill processes taking up GPU:**
Run `fuser -k /dev/nvidia0`.

**Inspect HTML from the dataset:**
Run `pnpx -y http-server` and navigate to the page through web browser.

**Debug CPython code:**
Run `gdb -ex=r --args python <path_to_script>.py`.
After a crash, issue GDB command `backtrace`.

**Get HTML from a scraping log:**
For example, if it's on line 11, run
`sed '11!d' data/scraping-logs/2022-03-21T13-26-38.056Z.txt
| jq -r '.html' > data/page.html`
(or `.visuals > data/visuals.json` or `.screenshot > data/screenshot.txt`).

**Start Jupyter server manually (e.g., in Gitpod):**
Run `jupyter notebook --allow-root --no-browser --NotebookApp.token=''
--NotebookApp.disable_check_xsrf=True --notebook-dir="$(pwd)"`.
