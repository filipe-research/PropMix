import matplotlib.pyplot as plt
import numpy as np


def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins

def plot_gmm_remove_noisy(path_plot, min_margin, idx_guess_correct, idx_guess_wrong, idx_keep):

	bins = compute_histogram_bins(min_margin, 0.01)
	plt.hist(np.array(min_margin),bins=bins, color='k', alpha=0.5, label='preds (%d)'%(len(min_margin)))
	plt.hist(np.array(min_margin)[idx_guess_correct],bins=bins, color='b', alpha=0.5, label='correct pred (%d)'%(len(idx_guess_correct)))
	plt.hist(np.array(min_margin)[idx_guess_wrong],bins=bins, color='r', alpha=0.5, label='wrong pred (%d)'%(len(idx_guess_wrong)))
	plt.hist(np.array(min_margin)[idx_keep],bins=bins, color='g', alpha=0.5, label='remove (%d)'%(len(idx_keep)))
	plt.xlabel('conf')
	plt.legend()
	plt.savefig(path_plot)
	plt.clf()


def plot_histogram_loss(path_plot, all_loss, inds_clean, inds_noisy):

	num_inds_clean = len(inds_clean)
	num_inds_noisy = len(inds_noisy)
	perc_clean = 100*num_inds_clean/float(num_inds_clean+num_inds_noisy)

	data = all_loss[0][-1].numpy()
	bins = compute_histogram_bins(data, 0.01)
	plt.hist(data[inds_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean - %d (%.1f%%)'%(num_inds_clean,perc_clean))
	if len(inds_noisy) >0:
		plt.hist(data[inds_noisy], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy- %d (%.1f%%)'%(num_inds_noisy,100-perc_clean))
	plt.xlabel('loss');
	plt.ylabel('number of data')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

	# plt.grid()
	#plt.savefig('%s/sep_loss_epoch%03d.png' % (path_exp,epoch))
	plt.savefig(path_plot)
	plt.clf()


def plot_modelview_histogram_loss(path_plot, all_loss, inds_clean, inds_noisy, idx_view_labeled, idx_view_unlabeled):
	num_inds_clean = len(inds_clean)
	num_inds_noisy = len(inds_noisy)

	missed_clean = np.asarray([i for i in inds_clean if i not in idx_view_labeled])
	missed_noisy = np.asarray([i for i in inds_noisy if i not in idx_view_unlabeled])

	num_view_labeled = len(idx_view_labeled)
	num_view_unlabeled = len(idx_view_unlabeled)
	total = num_view_labeled + num_view_unlabeled

	data = all_loss[0][-1].numpy()
	bins = compute_histogram_bins(data, 0.01)

	plt.hist(data[idx_view_labeled],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_clean(%d| %.1f%%)'%(num_view_labeled,100*num_view_labeled/float(total)))
	plt.hist(data[idx_view_unlabeled], bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, label='alg_view_noisy(%d| %.1f%%)'%(num_view_unlabeled,100*num_view_unlabeled/float(total)))
	plt.hist(data[missed_clean],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='#fb8072', label='FN (%d| %.1f%%)'%(len(missed_clean),100*len(missed_clean)/float(len(inds_clean))))
	if len(inds_noisy) >0:
		plt.hist(all_loss[0][-1].numpy()[missed_noisy],bins=bins, range=(0., 1.), edgecolor='black', alpha=0.5, color='k',label='FP (%d| %.1f%%)'%(len(missed_noisy),100*len(missed_noisy)/float(len(inds_noisy))))
	plt.xlabel('loss');
	plt.ylabel('number of data')
	plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

	# plt.grid()
	# plt.savefig('%s/view_sep_loss_epoch%03d.png' % (path_exp,epoch))
	plt.savefig(path_plot)
	plt.clf()



    



