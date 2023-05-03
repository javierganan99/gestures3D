import cv2
import sys, os, termios, tty, signal
import yaml


def load_yaml(file):
    """
    This function loads a yaml file
    """
    assert os.path.exists(file), "File not found in path {}".format(file)
    with open(file, "r") as f:
        params = yaml.safe_load(f)
    return params


def ensure_exist(path):
    """
    This function checks if path exist else create it
    """
    separated = path.split("/")
    exists = True
    for f in range(len(separated)):
        path = os.path.join(*separated[: f + 1])
        if not os.path.exists(path):
            os.mkdir(path)
            exists = False
    return exists


def images_to_video(input_folder_path, format=".png", name="plots.avi", fps=30):
    """
    This function receives a folder path containing images
    and transform them to a video, also deleting the containing folder
    """
    png_files = [f for f in os.listdir(input_folder_path) if f.endswith(format)]
    png_files.sort()
    first_image = cv2.imread(os.path.join(input_folder_path, png_files[0]))
    height, width, layers = first_image.shape
    output_video_path = os.path.dirname(input_folder_path)
    video = VideoSaver(output_video_path, size=(width, height), fps=fps, color=True, name=name)
    for png_file in png_files:
        image_path = os.path.join(input_folder_path, png_file)
        image = cv2.imread(image_path)
        video(image)
        os.remove(image_path)
    del video
    os.rmdir(input_folder_path)


class VideoSaver:
    """
    This class is used to save a video of specified frame size and fps
    """

    def __init__(self, path, size=(640, 512), fps=30, color=True, name="video.avi"):
        self.path = os.path.join(path, name)
        self.out = cv2.VideoWriter(
            self.path,
            apiPreference=0,
            fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
            fps=fps,
            frameSize=size,
            isColor=color,
        )

    def __call__(self, image):
        self.out.write(image)

    def __del__(self):
        self.out.release()


class StartKey(Exception):
    pass


class EndKey(Exception):
    pass


class HandleKeys:
    """
    This class is used for controlling the start-stop of a process with start/stop keys.
    main_class is the class where the process runs, and class_parameters is a dictionary
    containing its parameters. main_class should contain an accesible flag variable
    indicating whether its process should be running. The flag variable is turned on/off
    by the start/end keys.
    """

    def __init__(self, main_class, class_parameters, start_key="s", end_key="q"):
        self.main_class = main_class
        self.parameters = class_parameters
        self.start_key = start_key
        self.end_key = end_key

    def main_function(self):
        self.object = self.main_class(**self.parameters)
        try:
            while True:
                try:
                    if not self.object():
                        break
                except StartKey:
                    self.object.flag = True
                    print("Started!")
                except EndKey:
                    self.object.flag = False
                    print("Ended!")

        # Handle SIGINT from parent process
        except KeyboardInterrupt:
            print("Ending process...")
        return 0

    def __call__(self):
        # Help
        print("Press {0} to start the activity, {1} to end, or CTRL-C to abort".format(self.start_key, self.end_key))

        # Get file descriptor for stdin. This is almost always zero.
        stdin_fd = sys.stdin.fileno()

        # Fork here
        pid = os.fork()

        ###################### CHILD #####################
        if not pid:
            os.setsid()

            # Define signal handler for SIGUSR1 and SIGUSR2
            def on_signal(signum, frame):
                if signum == signal.SIGUSR1:
                    raise StartKey
                elif signum == signal.SIGUSR2:
                    raise EndKey

            # Catch SIGUSR1 and SIGUSR2
            signal.signal(signal.SIGUSR1, on_signal)
            signal.signal(signal.SIGUSR2, on_signal)

            # Now do the thing
            return self.main_function()

        ###################### PARENT ####################
        def on_sigchld(signum, frame):
            assert signum == signal.SIGCHLD
            sys.exit(0)

        # Catch SIGCHLD
        signal.signal(signal.SIGCHLD, on_sigchld)

        # Remember the original terminal attributes
        stdin_attrs = termios.tcgetattr(stdin_fd)

        # Change to cbreak mode, so we can detect single keypresses
        tty.setcbreak(stdin_fd)

        try:
            while True:
                # Wait for a keypress
                char = os.read(stdin_fd, 1).decode("utf-8")

                # If it was start_key, send SIGUSR1 to the child
                if char.lower() == self.start_key:
                    os.kill(pid, signal.SIGUSR1)

                # If it was end_key, send SIGUSR2 to the child
                if char.lower() == self.end_key:
                    os.kill(pid, signal.SIGUSR2)

        # Parent caught SIGINT - send SIGINT to child process
        except KeyboardInterrupt:
            os.kill(pid, signal.SIGINT)

        # Catch system exit
        except SystemExit:
            pass

        # Ensure we reset terminal attributes to original settings
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_attrs)

        # Return success
        print("Done!")
        return 0
