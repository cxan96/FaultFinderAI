package clasDC.factories;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;
import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import utils.FaultUtils;

/**
 * 
 * @author m.c.kunkel <br>
 *         a CLASDriftChamber contains 2 superlayers 1-2, 3-4, 5-6. Each
 *         superlayer contains 6 layers and 112 wires. <br>
 *         FaultFactory provides the layers and wire faults.
 * 
 */

public class CLASDriftChamber extends CLASComponent {
	private int region;

	@Getter(AccessLevel.NONE)
	private Map<Integer, Pair<Image, List<Fault>>> dcFaults = null;

	@Getter(AccessLevel.NONE)
	private CLASSuperlayer aSuperlayer = null;

	@Builder
	public CLASDriftChamber(int region, int nchannels, int maxFaults, List<FaultNames> desiredFaults,
			boolean singleFaultGen) {
		if (region > 3 || region < 1) {
			throw new IllegalArgumentException("Invalid input: (region), must have values less than"
					+ " (4) and more than (0). Received: (" + region + ")");
		}
		this.region = region;
		this.nchannels = nchannels;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;
		init();
		concat();
	}

	private void init() {
		this.dcFaults = new HashMap<>();
		this.aSuperlayer = CLASSuperlayer.builder().superlayer(1).nchannels(nchannels).maxFaults(this.maxFaults)
				.desiredFaults(desiredFaults).singleFaultGen(singleFaultGen).build();
		if (this.region == 1) {
			dcFaults.put(1, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));
			aSuperlayer = aSuperlayer.getNewSuperLayer(2);
			dcFaults.put(2, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));

		} else if (this.region == 2) {
			aSuperlayer = aSuperlayer.getNewSuperLayer(3);
			dcFaults.put(1, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));
			aSuperlayer = aSuperlayer.getNewSuperLayer(4);
			dcFaults.put(2, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));

		} else {
			aSuperlayer = aSuperlayer.getNewSuperLayer(5);
			dcFaults.put(1, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));
			aSuperlayer = aSuperlayer.getNewSuperLayer(6);
			dcFaults.put(2, Pair.of(aSuperlayer.getImage(), aSuperlayer.getFaultList()));

		}

	}

	private void concat() {
		// if (this.region == 1) {
		INDArray a = dcFaults.get(1).getLeft().getImage();
		INDArray b = dcFaults.get(2).getLeft().getImage();
		if (a.rank() != b.rank() || a.size(a.rank() == 3 ? 1 : 2) != b.size(b.rank() == 3 ? 1 : 2)
				|| a.size(a.rank() == 3 ? 2 : 3) != b.size(b.rank() == 3 ? 2 : 3)) {
			throw new IllegalArgumentException("Invalid input: arrays are not of equal rank in addImages()");
		}
		/**
		 * Concat along the rows i.e layers
		 */
		INDArray ret = Nd4j.concat(a.rank() == 3 ? 1 : 2, a, b);
		int rank = ret.rank();
		int rows = (int) ret.size(rank == 3 ? 1 : 2);
		int cols = (int) ret.size(rank == 3 ? 2 : 3);
		int nchannels = (int) ret.size(rank == 3 ? 0 : 1);

		this.image = new Image(ret, nchannels, rows, cols);

		/**
		 * This will modify and append the Fault list
		 */
		List<Fault> aList = dcFaults.get(1).getRight();
		List<Fault> bList = dcFaults.get(2).getRight();

		for (Fault fault : bList) {
			fault.offsetFaultCoodinates(6.0, "y");

			aList.add(fault);
		}
		this.faultList = aList;

	}

	public static void main(String[] args) {
		// CLASDriftChamber c = new CLASDriftChamber(1, 1, 3);
		CLASDriftChamber c = CLASDriftChamber.builder().region(1).nchannels(1).maxFaults(5)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE, FaultNames.PIN_SMALL)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();
		INDArray ret = c.getImage().getImage();
		FaultUtils.draw(c.getImage());

		int rank = ret.rank();
		int rows = (int) ret.size(rank == 3 ? 1 : 2);
		int cols = (int) ret.size(rank == 3 ? 2 : 3);
		int nchannels = (int) ret.size(rank == 3 ? 0 : 1);
		// System.out.println(rank + " " + rows + " " + cols + " " + nchannels);
		System.out.println(ret.shapeInfoToString());

		// for (Fault fault : c.getFaultList()) {
		// fault.printWireInformation();
		// }

	}

}
