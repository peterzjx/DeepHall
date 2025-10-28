import signal

class GracefulKiller:
    """Capture SIGINT and SIGTERM so that we can save checkpoints before exit."""

    kill_now = False

    def __init__(self):
        self.original_int = signal.signal(signal.SIGINT, self.exit_gracefully)
        self.original_term = signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Mark as exit and restore signal handlers."""
        del signum, frame
        if self.kill_now:  # Only handle the first signal
            return
        print("\r", end="")  # Clear ^C
        signal.signal(signal.SIGINT, self.original_int)
        signal.signal(signal.SIGTERM, self.original_term)
        self.kill_now = True