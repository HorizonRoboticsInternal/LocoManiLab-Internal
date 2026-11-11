"""RoboOrchard Revo2 inference on the shared evaluation runtime."""
from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register


@register('roboorchard')
class RoboOrchardPlugin(ModelPlugin):
    def _sim_server(self, *, stage, prompt, port, backend='holomotion', run_dir=None, **opts):
        if stage != 'eval' or backend != 'holomotion':
            return StageResult(ok=False, message='RoboOrchard currently requires eval --backend holomotion')
        from locomanilab.runtime import script_path
        job = self.ctx.input_path(run_dir) if run_dir else self.ctx.path('roboorchard_job')
        if job is None:
            return StageResult(ok=False, message='eval: set paths.roboorchard_job or --run-dir to a saved training job')
        if 'roboorchard' not in self.ctx.repos:
            return StageResult(ok=False, message='eval: configure repos.roboorchard')
        repo = self.repo('roboorchard')
        if not self.runner.dry_run and not job.is_dir():
            return StageResult(ok=False, message=f'eval: saved job directory does not exist: {job}')
        checkpoint = self.manifest.deploy.get('checkpoint')
        runtime = self.ctx.path('roboorchard_runtime', 'runs/roboorchard/runtime')
        cmd = ['python', str(script_path('roboorchard/server.py')), '--repo', str(repo), '--job', str(job),
               '--runtime-root', str(runtime), '--port', str(port)]
        if checkpoint:
            cmd.extend(["--checkpoint", str(checkpoint)])
        return SimServer(self.server_argv(cmd, cwd=repo), repo, prompt=prompt, inference_rate=3.125)
