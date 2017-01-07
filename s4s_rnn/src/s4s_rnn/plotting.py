import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def plot_multi_y(y1, y2, x_label, y1_label, y2_label, title, x=None):
    if x is None:
        x = list(range(max(len(y1), len(y2))))
        pass
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'r.')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
        pass

    ax2 = ax1.twinx()
    ax2.plot(x, y2, '--')
    ax2.set_ylabel(y2_label, color='b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
        pass
    plt.title(title)
    plt.grid()
    plt.show()
    return


def plot_inputs(inputs):
    """
    Function to plot input features, each feature in a separated subplot
    :param inputs: input features as (num_samples, num_features) array
    :return: None
    """
    num_features = inputs.shape[1]
    feature_names = ['distance', 'velocity', 'acceleration', 'time']
    figure, subplots = plt.subplots(num_features, sharex=True)
    for i in range(num_features):
        subplots[i].plot(inputs[:, i], '-or', label=feature_names[i])
        subplots[i].set_title('Plot of normalized %s' % feature_names[i])
        max_input = np.max(inputs[:, i])
        min_input = np.min(inputs[:, i])
        difference = max_input - min_input
        subplots[i].set_ylim([min_input - 0.1*difference,
                              max_input + 0.1*difference])
        subplots[i].grid()
        pass

    figure.subplots_adjust(hspace=0.2)
    figure.set_size_inches((10, 10))
    plt.setp([a.get_xticklabels() for a in figure.axes[:-1]], visible=False)
    plt.show()
    return


def plot_predictions(predictions, prediction_names, true_output, title, file_name=None,
                     y_label="Heart rate (hbm)", x_label="Time steps", show_plot=True):
    """
    Visualise comparison between prediction and actual data

    :param predictions: list of predicted outputs
    :param prediction_names: names of predictions for plot labels
    :param true_output: actual outputs
    :param file_name: name of image file for saving plot
    :param title:
    :param y_label:
    :param x_label:
    :param save_plot: if True will write plot image to file_name
    :param show_plot: if True will show plot
    :return: None
    """
    if len(predictions) != len(prediction_names):
        print("Lengths of prediction list and prediction names must equal")
        return

    max_len = max(map(lambda pred : len(pred), predictions))
    x = list(range(max_len))

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    colors = cycle('rbgcmykw')
    lines = []
    line_actual, = plt.plot(x, true_output[-max_len:], '-o', c=next(colors), markersize=4,
                            label='True output')
    lines.append(line_actual)
    for index, prediction in enumerate(predictions):
        line_predict, = plt.plot(x[-len(prediction):], prediction, '-+',
                                 c=next(colors), markersize=4,
                                 label=prediction_names[index])
        lines.append(line_predict)
        pass

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(handles=lines, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fancybox=True,
              shadow=True, ncol=2)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()

    if type(file_name).__name__ == 'str':
        plt.savefig(file_name)
        pass

    if show_plot:
        plt.show()
        pass
    pass


def box_plot_error(abs_errors, title, labels):
    """

    :param abs_errors:
    :param title:
    :param labels:
    :return:
    """
    fig, ax1 = plt.subplots(figsize=(10, 7))
    fig.canvas.set_window_title(title)
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = plt.boxplot(abs_errors, showmeans=True, labels=labels, notch=0, sym='+', vert=1, whis=1.5)

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Squared Error')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, len(labels) + 0.5)
    top = max([np.max(t) for t in abs_errors])
    bottom = min([np.min(t) for t in abs_errors])
    difference = top - bottom
    ax1.set_ylim(bottom - 0.10*difference, top + 0.08*difference)
    xtick_names = plt.setp(ax1, xticklabels=labels)
    plt.setp(xtick_names, rotation=45, fontsize=8)

    # Add upper X-axis tick labels with the mse
    pos = np.arange(len(labels)) + 1
    upper_labels = [str(np.round(np.sqrt(np.mean(s**2)), 2)) for s in abs_errors]
    for tick, label in zip(range(len(labels)), ax1.get_xticklabels()):
        ax1.text(pos[tick], top + 0.02*difference, upper_labels[tick],
                 horizontalalignment='center', size='x-small', weight='bold')
        pass

    # plt.yscale('log', basey=2)
    plt.show()
    return


def bar_plot_error(squared_error_groups, title, prediction_labels, group_names):
    """

    :param squared_error_groups:
    :param title:
    :param prediction_labels:
    :param group_names:
    :return: None
    """
    colors = cycle('rbgcmykw')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title(title)
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    num_group = len(squared_error_groups)
    width = 1.0 / (num_group * (len(squared_error_groups[0]) + 1))
    location = np.arange(num_group)
    rect_groups = []
    for prediction_index in range(len(squared_error_groups[0])):
        means = []
        stds = []
        for group_index in range(num_group):
            means.append(np.mean(squared_error_groups[group_index][prediction_index]))
            stds.append(np.std(squared_error_groups[group_index][prediction_index]))
            pass
        print(len(means))
        rects = ax.bar(location + prediction_index*width, means, width, color=next(colors), yerr=stds)
        rect_groups.append(rects[0])
        pass

    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_xticks(location + width*num_group/2.0)
    ax.legend(rect_groups, group_names)

    xtick_names = plt.setp(ax, xticklabels=prediction_labels)
    plt.setp(xtick_names, rotation=45, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Squared Error')
    plt.show()
    return
