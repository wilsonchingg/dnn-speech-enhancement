from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard

def get_callbacks(filename, early_stop = -1):
	best_checkpointer = ModelCheckpoint(filepath= filename + '_best.hdf5',
	verbose=2, save_best_only=True, period=1)
	csv_logger = CSVLogger(filename + '.log', append=True)
	cbs = [best_checkpointer, csv_logger]
	if early_stop >= 0:
		cbs.append(EarlyStopping(monitor='val_loss', patience=early_stop))
	return cbs
