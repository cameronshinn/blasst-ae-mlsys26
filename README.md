# BLASST Artifact Evaluation for MLSys 2026

Start the docker container and clone with external subodules. If you are using a cloud service that pre-loads a docker image, you can point it to [nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc3)

FIXME: Double check where start_docker.sh mounts the repo to
```
git clone git@github.com:cameronshinn/blasst-ae-mlsys26.git --recursive
cd blasst-ae-mlsys26 && ./start_docker.sh
cd ~  # We tested on the home directory since /workspace was mounted to a slow network drive
```

The artifacts are split among prefill/decode and Hopper/Blackwell (4 folders). Each folder has instructions how to reproduce its results as well as expected numbers.
