"""DQN test harness for predefined scenarios."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.Driver_dqn import DriverDQN
class driver_test(DriverDQN):
    """DriverDQN wrapper for test-case evaluation and recording."""

    def start_screen_recording(
        self,
        filename: str = "videos/test_cases_recording.mp4",
        fps: int = 30,
        resolution: str = "1920x1080",
    ):
        # Start screen recording with ffmpeg in a subprocess
        command = [
            'ffmpeg', '-y', '-f', 'x11grab',  
            '-framerate', str(fps),
            '-video_size', resolution,
            '-i', ':1.0',  # You can specify a window instead, e.g. ':0.0+0,0'
            '-codec:v', 'libx264', filename
        ]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop_screen_recording(self) -> None:
        self.recording_process.terminate()  # Stop the ffmpeg subproces
    
    def test_spesific_cases(self, render: bool = True) -> None:
        self.q_network.load_state_dict(torch.load("dqn.pt"))
        self.q_network.eval()
        self.epsilon = 0
        rewards = []
        
        self.recording_process = self.start_screen_recording("videos/test_cases_recording.mp4", fps=30)
        num_of_test_cases = len(self.env.lane_manager.test_cases)
        self.env.effective_sim_length = 5000
        self.env.effective_sim_time = 100
        self.env.lane_manager.num_of_vehicles = 0
        for case_num in range(num_of_test_cases):
            state, test_case_name = self.env.reset_for_test_cases()
            self.env.sim_start_time = time.time()
            done = False
            total_reward = 0
            speeds = []
            times = []
            wanted_speeds = []
            start_time = time.time()

            # Create a live plot
            plt.ion()
            fig, ax = plt.subplots()
            line_speed, = ax.plot([], [], "r-", label="Ego Speed")
            line_wanted_speed, = ax.plot([], [], "b-", label="Ego Wanted Speed")
            ax.set_xlim(0, self.env.effective_sim_time)  
            ax.set_ylim(0, 200)  
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed (km/h)')
            ax.set_title(f"{test_case_name}")
            ax.legend()
            fig.canvas.manager.window.move(1200,600)
            previous_action = 0
            while not done:

                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action, previous_action)
                total_reward += reward
                state = next_state
                previous_action = action
                # Record the speed and time
                current_time = time.time() - start_time
                speeds.append(self.env.lane_manager.ego_vehicle.speed * 36 / 10)
                wanted_speeds.append(self.env.lane_manager.ego_vehicle.v_max * 36 / 10)
                times.append(current_time)

                # Update live plot
                line_speed.set_xdata(times)
                line_speed.set_ydata(speeds)
                
                line_wanted_speed.set_xdata(times)
                line_wanted_speed.set_ydata(wanted_speeds)

                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

                if render:
                    #self.env_plotter.update_highway(self.env)
                    self.env_plotter.render()

            rewards.append(total_reward)

            # Save the speed plot after the test case ends
            plt.ioff()  # Turn off interactive mode
            plt.savefig(f"images/test_case_{case_num}_speed_plot.png")
            plt.close(fig)  # Close the figure
        self.stop_screen_recording()
        time.sleep(1)


def speed_up_video(
    input_file: str = "videos/test_cases_recording.mp4",
    output_file: str = "videos/output.mp4",
) -> None:
    # The ffmpeg command to speed up the video
    command = [
        'ffmpeg', '-i', input_file, 
        '-filter:v', 'setpts=0.2*PTS',  # Adjust the playback speed (0.5 is 2x speed)
        output_file
    ]
    
    # Run the command and capture output and error messages
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)  # Print any standard output from the command
        print(result.stderr)  # Print any error messages from the command

        if result.returncode != 0:
            print(f"Error running ffmpeg command: {result.stderr}")
        else:
            print(f"Video successfully saved as {output_file}")

    except Exception as exc:
        print(f"Exception occurred: {str(exc)}")
if __name__ == "__main__":
    with open("config_dqn.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    driver = driver_test(config)
    #driver.train(num_episodes= 4000,render=False) 
    #driver.test(render=True)
    driver.test_spesific_cases(render=True)
    #speed_up_video("videos/test_cases_recording.mp4", "videos/output.mp4")
