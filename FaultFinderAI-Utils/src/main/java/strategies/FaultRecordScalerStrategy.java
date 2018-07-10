package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface FaultRecordScalerStrategy {

	public void normalize(INDArray features);
}
