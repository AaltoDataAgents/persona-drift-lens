# Persona Drift Lens on Noppe

A teaching demonstration that shows the inner workings of a small language model while it writes:
how close each reply sits to the model's default Assistant, and which voice it sounds most like. It
runs on the processor, in a session on [Noppe](https://noppe.csc.fi) (CSC notebooks).

This repository holds only what a Noppe session needs. Noppe clones it when a session starts. It is
generated from the demonstration's main repository, so changes should be made there.

## Setting it up on Noppe

1. **Build the image.** In the workspace's custom image builder, choose a Jupyter base image and add
   the pip package `jupyter-server-proxy==4.6.0`.
2. **Create the application:**
   - **Container image:** the image from step 1
   - **Session memory:** 5 GiB
   - **Download method:** git clone, with the address of this repository
   - **Environment variable:** `JUPYTER_CONFIG_PATH=/home/jovyan/persona-drift-lens/deploy/noppe/conda`
   - **Shared folder:** on
3. **Fill the shared folder.** Start one session as the workspace owner, open a terminal and run the
   following. It installs the Python environment and downloads the model (about 2 GB in all), once
   for all participants:

       bash ~/persona-drift-lens/deploy/noppe/conda/setup-shared.sh

4. **Check it.** Delete that session and start a new one. It should open on the demonstration, which
   takes a little while to load the model.

## Credit and licences

By Emrecan Gülay, Aalto University. This is a selection of files from the main repository; see
`LICENSE.md`. The code is under the MIT licence, and the data under CC BY 4.0. The language model,
Qwen3-1.7B, is not included: it is downloaded from Hugging Face and keeps its own licence.
