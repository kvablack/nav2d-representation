import tensorboard_reducer as tbr
import matplotlib.pyplot as plt
import glob

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:grey', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
labels = ['online/neweps_e1.0', 'online/neweps_e0.5', 'online/neweps_e0.0', 'online/neweps_e0.1',
            'online/neweps_convenc_e1.0_nogoal']
          # 'neweps_e0.5_nogoal_online', 'neweps_e0.0_nogoal_online', 'neweps_e0.1_nogoal_online', 'neweps_e1.0_nogoal_online', 'baseline']

for label, color in zip(labels, colors):
    dirs = glob.glob(f"experiments/{label}/*")
    events_dict = tbr.load_tb_events(dirs, strict_steps=False, min_runs_per_step=1)
    success_rate = events_dict['success_rate']
    success_rate = success_rate.interpolate(method="index")
    mean = success_rate.mean(axis=1)
    std = success_rate.std(axis=1)
    plt.plot(mean, label=label, color=color)
    plt.fill_between(mean.index, mean - std, mean + std, facecolor=color, alpha=0.5)

plt.legend()
plt.xlabel("timesteps")
plt.ylabel("success rate")
plt.show()
