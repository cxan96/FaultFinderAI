/**
 * 
 */
package clasDC.factories;

import java.util.List;

import org.datavec.image.data.Image;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 */
public abstract class CLASComponent implements CLASFactory {
	protected int nchannels;
	protected int maxFaults;
	protected List<FaultNames> desiredFaults = null;
	protected boolean singleFaultGen;
	@Getter
	protected Image image = null;
	@Getter
	protected List<Fault> faultList = null;

}
