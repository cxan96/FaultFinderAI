package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MinMaxStrategy implements FaultRecordScalerStrategy {

	private double percent;

	private double minRange;
	private double maxRange;

	public MinMaxStrategy() {
	}

	public MinMaxStrategy(double percent) {
		this.percent = percent;
	}

	@Override
	public void normalize(INDArray features) {
		int rank = features.rank();
		if (rank == 2) {
			applyMinMax(features);
		} else {
			int nchannels = (int) features.size(rank == 3 ? 0 : 1);
			int rows = (int) features.size(rank == 3 ? 1 : 2);
			int cols = (int) features.size(rank == 3 ? 2 : 3);

			if (nchannels == 1) {
				applyMinMax(features);
			} else if (nchannels == 3) {
				features.divi(255); // Scaled to 0->1

			} else {
				throw new ArithmeticException("Number of channels must be 1 or 3");
			}
		}

	}

	private void applyMinMax(INDArray features) {

		double maxRange = (double) features.maxNumber();
		// System.out.println("Before scaling max = " + maxRange);
		double minRange = (double) features.minNumber();
		double minMaxBias = percent * (maxRange - minRange);
		if (minRange != 0)
			features.subi(minRange); // Offset by minRange
		features.divi((2.0 * minMaxBias + maxRange - minRange));
		if ((double) features.maxNumber() < 1) {
			features.addi(1. - (double) features.maxNumber());
		}
		if ((double) features.maxNumber() > 1) {
			System.out.println(features.maxNumber() + " WHAT!!!! We scaled to a number greater than 1  maxNumber");
		}
	}

}
