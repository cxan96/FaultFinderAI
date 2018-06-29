package testing;

import faulttypes.*;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.util.NDArrayUtil;

public class FaultExplorer {
    public static void main (String args []) {
	
	FaultFactory factory = new FaultFactory();
	FaultData deadWire = factory.getFault(5);

	
	INDArray labelArray = NDArrayUtil.toNDArray(factory.getReducedLabel());
	//                                          ^^^^^^^^^^^^^^^^^^^^^^^^^
	
	System.out.println(labelArray);
    }
}
