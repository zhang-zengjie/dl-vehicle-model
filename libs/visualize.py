import matplotlib.pyplot as plt
import os

figure_dir = 'figures'


def visualize_test_1D_distribution(labels, predictions):

    plt.figure(figsize=(5, 5))
    for i in range(len(labels)):
        plt.plot(labels[i][:, 0, 0], predictions[i][:, 0, 0], "bo")
    plt.plot([-0.5, 5.5], [-0.5, 5.5], 'r', linewidth=2, linestyle='-.', label='Zero error')
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Ground Truth", fontsize=12)
    plt.ylabel("Prediction", fontsize=12)
    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 5.5) 
    plt.legend()
    
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(os.path.join(figure_dir, 'dl_model_test.svg'), format='svg')
    plt.show()


def visualize_test_2D_trajectory(sample, label, prediction):

    plt.figure(figsize=(8, 2))
    plt.plot(sample[0], sample[1], "k", linewidth=2, label="History")
    plt.plot(label[0], label[1], "r", linewidth=2, label="Ground Truth")
    plt.plot(prediction[0], prediction[1], "b", linewidth=2, label="Prediction")

    plt.grid(True)
    #plt.axis("equal")
    plt.xlabel("$x$ (m)", fontsize=12)
    plt.ylabel("$y$ (m)", fontsize=12)
    plt.legend()

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.subplots_adjust(top=0.88, bottom=0.272, left=0.076, right=0.996)
    plt.savefig(os.path.join(figure_dir, 'dl_prediction_test.svg'), format='svg')
    plt.show()