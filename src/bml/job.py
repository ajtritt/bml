import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


class JobWriter:

    def __init__(self, config, job_time=240, inputs_name="inputs", job_prefix="ferroX_"):
        """
            Args:
                config                  : the config file
                job_prefix (str)   : a prefix to append to job_name when submitting to Slurm
        """
        self.config = config
        self.exe_path = os.path.expandvars(config['exe_path'])
        self.project = config['nersc_project']
        self.job_time = job_time
        self.inputs_name = inputs_name
        self.job_prefix  = job_prefix

    def write_job(self, f, base_outdir, job_name):
        SCRIPT=f"""#!/bin/bash
#SBATCH -A {self.project}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t {self.job_time}
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -e {base_outdir}.%j/error.txt
#SBATCH -o {base_outdir}.%j/output.txt
#SBATCH -J {job_name}
cd {base_outdir}.$SLURM_JOB_ID
export SLURM_CPU_BIND="cores"
srun {self.exe_path} {self.inputs_name}"""
        print(SCRIPT, file=f)

    def write_array_job(self, f, base_outdir, job_name, n_tasks, start_task=0):
        SCRIPT=f"""#!/bin/bash
#SBATCH -A {self.project}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t {self.job_time}
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -e {base_outdir}/it%6a/error.txt
#SBATCH -o {base_outdir}/it%6a/output.txt
#SBATCH -J {job_name}
#SBATCH -a {start_task}-{start_task + n_tasks - 1}
printf -v it "%06d" $SLURM_ARRAY_TASK_ID
cd {base_outdir}/it$it
export SLURM_CPU_BIND="cores"
srun {self.exe_path} {self.inputs_name}"""
        print(SCRIPT, file=f)

    @classmethod
    def fmt_param(cls, param):
        """
        Format paramters to string representations that FerroX requires in the inputs file
        """
        if isinstance(param, list):
            return " ".join(cls.fmt_param(_) for _ in param)
        elif isinstance(param, (int, np.int64)):
            return str(param)
        elif isinstance(param, (float, np.float64)):
            if param == 0.0:
                return '0.0'
            return f'{param:0.3g}'
        elif isinstance(param, str):
            return param
        else:
            raise ValueError(f"Got value of type {type(param)}")

    def write_inputs(self, f, inputs):
        for k in inputs:
            print(f"{k} = {self.fmt_param(inputs[k])}", file=f)

    @staticmethod
    def submit_job(sh_path):
        cmd = f'sbatch {sh_path}'
        output = subprocess.check_output(
                    cmd,
                    stderr=subprocess.STDOUT,
                    shell=True).decode('utf-8')

        result = re.search('Submitted batch job (\d+)', output)
        if result is not None:
            jobid = int(result.groups(0)[0])
        else:
            print(f'Job submission failed: {output}', file=sys.stderr)
            jobid = -1

        print(f"Submitted job {jobid}", file=sys.stderr)
        if jobid == -1:
            exit()
        return jobid

    def submit_workflow(self, inputs, outdir, job_name, submit=True):
        """
        Prepare an inputs file and sbatch script for running the next step and submit job to Slurm.

        Args:
            inputs (dict)           : the value of the inputs to use
            job_name (str)          : the job name to use when submitting to Slurm
            submit (bool)           : submit job to Slurm if True, else print input and sbatch script to standard output
        """
        inputs_f = tempfile.NamedTemporaryFile('w', prefix='inputs', delete=False)
        inputs_path = inputs_f.name
        self.write_inputs(inputs_f, inputs)
        inputs_f.close()

        sh_f = tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False)
        sh_path = sh_f.name
        base_outdir = os.path.join(outdir, job_name)
        self.write_job(sh_f, base_outdir, job_name=f"{self.job_prefix}{job_name}")
        sh_f.close()

        jobid = None
        if submit:
            jobid = self.submit_job(sh_path)
            outdir = f"{base_outdir}.{jobid}"
        else:
            outdir = f"{base_outdir}.unsubmitted"

        os.makedirs(outdir, exist_ok=not submit)
        dest_sh = f"{outdir}/run.sh"
        dest_inputs = f"{outdir}/{self.inputs_name}"
        shutil.copyfile(sh_path, dest_sh)
        shutil.copyfile(inputs_path, dest_inputs)
        print(f"Copied submission script to {dest_sh}", file=sys.stderr)
        print(f"Copied inputs file to {dest_inputs}", file=sys.stderr)

        return outdir


class ChainJobWriter(JobWriter):

    def write_job(self, f, base_outdir, job_name):
        super().write_job(f, base_outdir, job_name)
        print(f"cd {os.getcwd()}", file=f)
        print(f"{os.path.basename(sys.argv[0])} {' '.join(sys.argv[1:])}", file=f)

