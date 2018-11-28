package clasDC.factories;

import java.util.List;

import org.datavec.image.data.Image;

import clasDC.faults.Fault;

public interface CLASFactory {

	Image getImage();

	List<Fault> getFaultList();

}
