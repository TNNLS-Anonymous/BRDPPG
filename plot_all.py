import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import Rbf


from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def load(path):
    record_dirs = os.listdir(path)
    records = []
    for record_dir in record_dirs:
        if not os.path.isdir(os.path.join(path, record_dir)):
            continue
        records_per_run = []
        for run_id in range(2):
            x_file = os.path.join(os.path.join(os.path.join(path, record_dir), 'eval'), 'x_{}.npy'.format(run_id))
            y_file = os.path.join(os.path.join(os.path.join(path, record_dir), 'eval'), 'y_{}.npy'.format(run_id))
            x = np.load(x_file)
            y = np.load(y_file)
            records_per_run.append((x,y))
        records.append(records_per_run)
    
    return records

def plot(records, labels, min_val, max_val, max_traj, B, save_path=None, val_type='plain', special=0):
    plt.clf()
    
    fig ,ax= plt.subplots(1,1,figsize=(8,4))
    for record, (label, color) in zip(records, labels):
        y_s = []
        for x, y in record:
            if val_type == 'plain':
                y_s.append(y[-1,:-1])
            elif val_type == 'linear':
                y_s.append(Rbf(x[-1,:-1], y[-1,:-1], function='linear')(np.arange(max_traj//B)))

        
        
        l, h = st.norm.interval(0.90, loc=np.mean(y_s, axis = 0), scale=st.sem(y_s, axis = 0))
            
        ax.plot(np.mean(y_s, axis = 0), label=label, color=color)
        ax.fill_between(range(int(max_traj//B)), l, h, alpha = 0.2, color=color)

    ax.set_ylim([min_val, max_val])
    
    ax.set_xlabel("Number of iterations", fontsize=16)
    ax.set_ylabel("Reward", fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.tick_params(labelsize=13) 

    if special != 0:
        #axins = inset_axes(ax, width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.1, 0.1, 1, 1), 
        # bbox_transform=ax.transAxes)
        if special == 1:
            axins = ax.inset_axes((0.4, 0.4, 0.4, 0.2))
        elif special == 2:
            axins = ax.inset_axes((0.45, 0.2, 0.3, 0.2))
        y_s = []
        label, color = labels[0]
        for x, y in records[0]:
            if val_type == 'plain':
                y_s.append(y[-1,:-1])
            elif val_type == 'linear':
                y_s.append(Rbf(x[-1,:-1], y[-1,:-1], function='linear')(np.arange(max_traj//B)))
        
        l, h = st.norm.interval(0.90, loc=np.mean(y_s, axis = 0), scale=st.sem(y_s, axis = 0))
            
        axins.plot(np.mean(y_s, axis = 0), label=label, color=color)
        axins.fill_between(range(int(max_traj//B)), l, h, alpha = 0.2, color=color)

        if special == 1:
            axins.set_xlim(200, 300)
            axins.set_ylim(-0.5, 0.2)
        elif special == 2:
            axins.set_xlim(150, 200)
            axins.set_ylim(9, 10)
        mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec='k', lw=1)
        axins.tick_params(labelsize=13)

    if save_path is not None:
        if val_type == 'plain':
            plt.savefig(os.path.join(save_path, 'plain_joint.png'))
        elif val_type == 'linear':
            plt.savefig(os.path.join(save_path, 'linear_joint.pdf'))

       


def main(envs=None):
    if envs == 'CartPole-v1':
        base_dirs = [
            "*",
            "*",
            "*",
            "*"
        ]
        record_labels = [
            [["BRDP-PG", "r"], ["DP-PG", "b"]],
            [['Single-Agent', "b"], ],
            [['DP-PG', 'b'], ["BRDP-PG", "r"]],
            [['DP-PG', 'b'], ["BRDP-PG", "r"]]
        ]
        change_order=[0,3]
        for i, (base_dir, labels ) in enumerate(zip(base_dirs, record_labels)):
            records = load(base_dir)
            if i in change_order:
                temp = records[0]
                records[0] = records[1]
                records[1] = temp
                temp = labels[0]
                labels[0] = labels[1]
                labels[1] = temp
                print(i)
            batch = 64
            max_traj = batch * 300
            if i!=6 and i != 4:
                plot(records, labels, 0, 600, max_traj, batch, base_dir)
                plot(records, labels, 0, 600, max_traj, batch, base_dir, val_type='linear')
            else:
                plot(records, labels, 0, 600, max_traj, batch, base_dir, special=2)
                plot(records, labels, 0, 600, max_traj, batch, base_dir, val_type='linear',special=2)
    if envs == 'HalfCheetah-v2':
        base_dirs = [
        "*",
        "*",
        "*",
        ]
        
        record_labels = [
            [['Single-Agent', "b"], ],
            [["BRDP-PG", "r"], ["DP-PG", "b"]],
            [["BRDP-PG", "r"], ["DP-PG", "b"]],
            [["BRDP-PG", "r"], ["DP-PG", "b"]],
        ]
        change_order=[1, 2, 3, 5]
        for i, (base_dir, labels ) in enumerate(zip(base_dirs, record_labels)):            
            records = load(base_dir)
            if i in change_order and i > 0:
                temp = records[0]
                records[0] = records[1]
                records[1] = temp
                temp = labels[0]
                labels[0] = labels[1]
                labels[1] = temp
                print(i)
            batch = 96
            max_traj = batch * 500

            if i!=5:
                plot(records, labels, -1000, 4000, max_traj, batch, base_dir)
                plot(records, labels, -1000, 4000, max_traj, batch, base_dir, val_type='linear')
            else:
                plot(records, labels, -1000, 4000, max_traj, batch, base_dir, special=1)
                plot(records, labels, -1000, 4000, max_traj, batch, base_dir, val_type='linear',special=1)



if __name__ == '__main__':
    main('CartPole-v1')
    main('HalfCheetah-v2')