"""PPO evaluation helper for test cases."""

from __future__ import annotations

import subprocess
import time

import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from environments.relative_to_ego_highway_env import HighwayEnv


class driver_test:
    def __init__(self) -> None:
        self.env: HighwayEnv = gym.make(id="highway_env")
        self.env.config.effective_sim_length = 5000
        self.env.config.effective_sim_time = 30
        
        self.env.config.num_of_vehicles = 0
        self.model = PPO.load("rl_scripts/tmp/best_model.zip")

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
        self.recording_process = self.start_screen_recording("videos/test_cases_recording.mp4", fps=30)
        num_of_test_cases = len(self.env.highway.lane_manager.test_cases)
        rewards = []
        
        for case_num in range(num_of_test_cases):
            _, _ = self.env.reset()
            obs, test_case_name = self.env.highway.reset_for_test_cases()
            #self.env.highway.lane_manager.sim_start_time = time.time()
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
            #line_wanted_speed, = ax.plot([], [], 'b-',label='Ego Wanted Speed')
            ax.set_xlim(0, self.env.config.effective_sim_time)
            ax.set_ylim(0, 200)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (km/h)")
            ax.set_title(f"{test_case_name}")
            ax.legend()
            #fig.canvas.manager.window.move(1200,600)
            previous_action = 0
            while not done:

                action, state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                current_time = time.time() - start_time
                speeds.append(self.env.highway.lane_manager.ego_vehicle.speed * 36 / 10)
                #wanted_speeds.append(self.env.highway.lane_manager.ego_vehicle.v_max * 36/10)
                times.append(current_time)

                # Update live plot
                line_speed.set_xdata(times)
                line_speed.set_ydata(speeds)
                
                #line_wanted_speed.set_xdata(times)
                #line_wanted_speed.set_ydata(wanted_speeds)

                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

                if render:
                    #self.env_plotter.update_highway(self.env)
                    self.env.render()

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
    driver = driver_test()
    
    #driver.train(num_episodes= 4000,render=False) 
    #driver.test(render=True)
    driver.test_spesific_cases(render=True)
    #speed_up_video("videos/test_cases_recording.mp4", "videos/output.mp4")
