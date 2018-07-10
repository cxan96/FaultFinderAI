package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public class StandardScore implements FaultRecordScalerStrategy {
	@Override
	public void normalize(INDArray features) {

		features.subi(features.meanNumber());
		features.divi(features.stdNumber());

	}
}
