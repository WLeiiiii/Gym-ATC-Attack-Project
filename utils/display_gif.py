import datetime

from matplotlib import pyplot as plt, animation

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def display_frames_as_gif(frames, atk):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    print("生成gif...")
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    if atk:
        anim.save('../logs/results/gifs/' + 'result_attack_' + current_time + '.gif', writer='pillow', fps=30)
        print("gif已保存！")
    else:
        anim.save('../logs/results/gifs/' + 'result_no_attack_' + current_time + '.gif', writer='pillow', fps=30)
        print("gif已保存！")
