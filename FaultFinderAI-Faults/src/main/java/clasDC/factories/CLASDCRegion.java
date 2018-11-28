package clasDC.factories;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import lombok.Builder;
import utils.FaultUtils;

/**
 * 
 * @author m.c.kunkel
 *
 *         A CLASDCRegion consists of 6 CLASDriftChambers
 * 
 */

public class CLASDCRegion extends CLASComponent {
	private int region;
	private Map<Integer, CLASDriftChamber> dcChambers = null;

	@Builder
	public CLASDCRegion(int region, int nchannels, int maxFaults, List<FaultNames> desiredFaults,
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
		this.dcChambers = new HashMap<>();
		this.faultList = new ArrayList<>();
		for (int i = 1; i < 7; i++) {
			CLASDriftChamber aChamber = CLASDriftChamber.builder().region(this.region).nchannels(this.nchannels)
					.maxFaults(this.maxFaults).desiredFaults(desiredFaults).singleFaultGen(singleFaultGen).build();
			dcChambers.put(i, aChamber);
		}
	}

	private void concat() {
		INDArray a = dcChambers.get(1).getImage().getImage();
		int rank = a.rank();
		int rows = (int) a.size(rank == 3 ? 1 : 2);
		int cols = (int) a.size(rank == 3 ? 2 : 3);
		int nchannels = (int) a.size(rank == 3 ? 0 : 1);
		INDArray ret = a;
		for (int i = 2; i < 7; i++) {
			CLASDriftChamber d = dcChambers.get(i);
			INDArray b = d.getImage().getImage();
			if (a.rank() != b.rank() || a.size(a.rank() == 3 ? 1 : 2) != b.size(b.rank() == 3 ? 1 : 2)
					|| a.size(a.rank() == 3 ? 2 : 3) != b.size(b.rank() == 3 ? 2 : 3)) {
				throw new IllegalArgumentException("Invalid input: arrays are not of equal rank in addImages()");
			}
			// ret = Nd4j.concat(2, ret, b);
			ret = Nd4j.concat(a.rank() == 3 ? 1 : 2, ret, b);
			for (Fault fault : d.getFaultList()) {
				fault.offsetFaultCoodinates(12.0 * (i - 1), "y");
				this.faultList.add(fault);
			}

		}
		rank = ret.rank();
		rows = (int) ret.size(rank == 3 ? 1 : 2);
		cols = (int) ret.size(rank == 3 ? 2 : 3);
		nchannels = (int) ret.size(rank == 3 ? 0 : 1);

		this.image = new Image(ret, nchannels, rows, cols);

	}

	public static void main(String[] args) {
		// CLASDCRegion factory = new CLASDCRegion(1, 3);
		CLASDCRegion factory = CLASDCRegion.builder().region(1).nchannels(1).maxFaults(20)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE, FaultNames.PIN_SMALL)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();

		INDArray ret = factory.getImage().getImage();

		FaultUtils.draw(factory.getImage());
		int rank = ret.rank();
		int rows = (int) ret.size(rank == 3 ? 1 : 2);
		int cols = (int) ret.size(rank == 3 ? 2 : 3);
		int nchannels = (int) ret.size(rank == 3 ? 0 : 1);
		System.out.println(rank + " " + rows + " " + cols + " " + nchannels);
		System.out.println(ret.shapeInfoToString());

		// for (Fault fault : factory.getFaultList()) {
		// fault.printWireInformation();
		// }
	}

}
