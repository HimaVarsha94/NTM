#import matplotlib.pyplot as plt
import csv



def parse_train_losses(filename):
	data = open(filename,'r')
	losses = []
	iterations = []
	cur_iter = 0
	leave = True
	for row in data:
		e = row.split(' ')
		if e[0] == 'iter':
			cur_iter = int(e[2])
			leave = True
			if int(e[2])%50==0:
				leave = False
		if e[0] == 'loss' and leave == False:
			# print cur_iter,float(e[2])
			# print "\n"
			losses.append(float(e[2]))
			iterations.append(cur_iter)

	return iterations,losses

def main():
	directory = "Recall/results"
	filename_gru = 'recall_test_gru_5k.log'
	filepath_gru = directory + "/" + filename_gru
	iterations_gru,losses_gru = parse_train_losses(filepath_gru)
	filename_lstm = 'recall_test_lstm_5k.log'
	filepath_lstm = directory + "/" + filename_lstm
	iterations_lstm, losses_lstm = parse_train_losses(filepath_lstm)
	with open(directory+'/'+'recall_5k.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile)
		for i in xrange(len(iterations_lstm)):
			spamwriter.writerow([iterations_lstm[i],losses_lstm[i], losses_gru[i]])

	# print iterations_lstm



if __name__ == '__main__':
	main()
