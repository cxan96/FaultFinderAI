package testing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Test {
	private static double yDiv = 5. / 6.;
	private static double xDiv = 111. / 112.;
	public static double[][] HVChannelPriors = { { 8.0 * xDiv, 6.0 * yDiv }, { 16.0 * xDiv, 6.0 * yDiv },
			{ 32.0 * xDiv, 6.0 * yDiv } };

	public static void main(String[] args) {

		// double[][] HVChannelPriors = { { 8.0, 6.0 }, { 16.0, 6.0 }, { 32.0, 6.0 } };

		double[][] divide = { { 2, 3 } };
		INDArray priors = Nd4j.create(HVChannelPriors);
		INDArray quot = Nd4j.create(divide);

		INDArray ret = quot.divi(priors);

		for (int i = 0; i < HVChannelPriors.length; i++) {
			for (int j = 0; j < HVChannelPriors[0].length; j++) {
				System.out.println(HVChannelPriors[i][j] + "  " + i + "  " + j + "  " + HVChannelPriors[i][j]);

			}
		}

	}

}
