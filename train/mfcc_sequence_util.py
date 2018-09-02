import numpy as np

def to_sequence_with_stride(bg_mfcc, left_pad=3, right_pad=3, stride_width=2):
	left_pad = left_pad * stride_width
	right_pad = right_pad * stride_width
	arr = []
	bg_mfcc = np.pad(bg_mfcc, ((left_pad, right_pad), (0, 0)), 'constant', constant_values=(0, 0))
	for i in range(left_pad, len(bg_mfcc)-right_pad):
		arr.append(bg_mfcc[i - left_pad: i + 1 + right_pad:stride_width])
	return arr

def to_sequence(bg_mfcc, left_pad=3, right_pad=3):
	arr = []
	bg_mfcc = np.pad(bg_mfcc, ((left_pad, right_pad), (0, 0)), 'constant', constant_values=(0, 0))
	for i in range(left_pad, len(bg_mfcc)-right_pad):
		arr.append(bg_mfcc[i - left_pad: i + 1 + right_pad])
	return arr
