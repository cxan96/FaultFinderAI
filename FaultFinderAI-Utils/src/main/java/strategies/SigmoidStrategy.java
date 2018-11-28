package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidStrategy implements FaultRecordScalerStrategy {
	@Override
	public void normalize(INDArray features) {

		new StandardScore().normalize(features);
		double[] dubs = new double[(int) features.length()];
		for (int i = 0; i < features.length(); i++) {
			double aDub = 1.0 / (1.0 + Math.exp(-features.getDouble(i)));
			dubs[i] = aDub;
		}
		INDArray create = Nd4j.create(dubs, new int[] { 1, dubs.length });
		// for (int i = 0; i < create.length(); i++) {
		// System.out.println("In Sigmoid " + create.getDouble(i));
		// }
		features.assign(create);

	}
}
