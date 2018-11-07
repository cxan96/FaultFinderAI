package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MinMaxStrategy implements FaultRecordScalerStrategy {

	private double percent;

	public MinMaxStrategy() {
		this.percent = 0.0;
	}

	public MinMaxStrategy(double percent) {
		this.percent = percent;
	}

	@Override
	public void normalize(INDArray features) {
		int rank = features.rank();
		int nchannels = features.size(rank == 3 ? 0 : 1);
		int rows = features.size(rank == 3 ? 1 : 2);
		int cols = features.size(rank == 3 ? 2 : 3);

		if (nchannels == 1) {
			applyMinMax(features);
		} else if (nchannels == 3) {
			INDArray a = Nd4j.create(nchannels, rows, cols);
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					double red = features.getDouble(0, 0, y, x);
					double green = features.getDouble(0, 1, y, x);
					double blue = features.getDouble(0, 2, y, x);

					double sum = red + green + blue;

					// features.putScalar(0, 0, y, x, red / sum * 255);
					// features.putScalar(0, 1, y, x, green / sum * 255);
					// features.putScalar(0, 2, y, x, blue / sum * 255);
					features.putScalar(0, 0, y, x, red / 255);
					features.putScalar(0, 1, y, x, green / 255);
					features.putScalar(0, 2, y, x, blue / 255);

				}
			}
			// features = a;
		} else {
			throw new ArithmeticException("Number of channels must be 1 or 3");
		}

	}

	private void applyMinMax(INDArray features) {
		double maxRange = (double) features.maxNumber();

		double minRange = (double) features.minNumber();
		double minMaxBias = percent * (maxRange - minRange);
		if (minRange != 0)
			features.subi(minRange); // Offset by minRange
		features.divi((2.0 * minMaxBias + maxRange - minRange));
	}

}
