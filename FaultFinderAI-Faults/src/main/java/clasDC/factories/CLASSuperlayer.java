/**
 * 
 */
package clasDC.factories;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import clasDC.faults.AbstractFaultFactory;
import clasDC.faults.FaultNames;
import clasDC.faults.SingleFaultFactory;
import lombok.Builder;
import utils.FaultUtils;

/**
 * @author m.c.kunkel <br>
 *         a CLASSuperlayer contains 6 layers and 112 wires. <br>
 *         FaultFactory provides the layers and wire faults.
 */

public class CLASSuperlayer extends CLASComponent {

	private AbstractFaultFactory factory = null;

	@Builder
	private CLASSuperlayer(int superlayer, int nchannels, int maxFaults, List<FaultNames> desiredFaults,
			boolean singleFaultGen) {
		if (superlayer > 6 || superlayer < 1) {
			throw new IllegalArgumentException("Invalid input: (superlayer), must have values less than"
					+ " ( 7) and more than (0). Received: (" + superlayer + ")");
		}
		this.nchannels = nchannels;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;

		this.factory = SingleFaultFactory.builder().superLayer(superlayer).maxFaults(maxFaults)
				.desiredFault(desiredFaults).randomSuperlayer(false).randomSmear(true).nChannels(nchannels)
				.singleFaultGen(singleFaultGen).build();
		this.image = factory.asUnShapedImageMatrix();
		this.faultList = factory.getFaultList();
	}

	public CLASSuperlayer getNewSuperLayer(int superLayer) {
		CLASSuperlayer sl = CLASSuperlayer.builder().superlayer(superLayer).nchannels(this.nchannels)
				.maxFaults(this.maxFaults).desiredFaults(this.desiredFaults).singleFaultGen(this.singleFaultGen)
				.build();
		return sl;

	}

	public static void main(String[] args) {
		CLASSuperlayer sl = CLASSuperlayer.builder().superlayer(1).nchannels(1).maxFaults(3)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE, FaultNames.PIN_SMALL)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();
		FaultUtils.draw(sl.getImage());

		sl = sl.getNewSuperLayer(6);
		FaultUtils.draw(sl.getImage());

		// for (Fault fault : sl.getFaultList()) {
		// fault.printWireInformation();
		// }
	}

}
