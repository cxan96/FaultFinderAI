/**
 * 
 */
package clasDC.factories;

import java.util.List;

import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 */
public abstract class CLASComponent implements CLASFactory {
	protected int nchannels;
	protected int minFaults;
	protected int maxFaults;
	@Getter
	protected FaultNames desiredFault;
	protected double desiredFaultGenRate;
	@Getter
	protected List<FaultNames> desiredFaults = null;
	protected boolean singleFaultGen;
	protected boolean isScaled = false;

	@Getter
	protected Image image = null;
	@Getter
	protected List<Fault> faultList = null;
	@Getter
	protected INDArray segmentationLabels = null;

}
