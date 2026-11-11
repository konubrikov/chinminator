#target photoshop
// Creates a native Scripts Action bound to this installed project location.
(function () {
    var script = new File(File($.fileName).parent.fsName + "/Chinminator-Retouch.jsx");
    if (!script.exists) throw new Error("Chinminator-Retouch.jsx is missing");
    var setName = "Chinminator Image Processor";
    try {
        var existing = new ActionReference(); existing.putName(charIDToTypeID("ASet"), setName);
        executeActionGet(existing);
        alert("Chinminator Image Processor is already installed. Select its Retouch action in Image Processor.");
        return;
    } catch (notInstalled) {}
    var root = script.parent.parent;
    var work = new Folder(root.fsName + "/work"); if (!work.exists) work.create();
    var parts = [];
    function number(value, count) {
        var bytes = "";
        for (var shift = count-1; shift >= 0; shift--) bytes += String.fromCharCode((value >>> (shift*8)) & 255);
        parts.push(bytes);
    }
    function unicode(value) {
        number(value.length+1,4);
        for (var i=0;i<value.length;i++) number(value.charCodeAt(i),2);
        number(0,2);
    }
    function ascii(value) { number(value.length,4); parts.push(value); }
    number(16,4); unicode(setName); number(1,1); number(1,4);
    number(0,2); number(0,1); number(0,1); number(0,2);
    unicode("Retouch"); number(1,1); number(1,4);
    number(0,1); number(1,1); number(0,1); number(0,1);
    parts.push("TEXT"); ascii("AdobeScriptAutomation Scripts"); ascii("Scripts");
    var descriptor = new ActionDescriptor();
    descriptor.putPath(charIDToTypeID("jsCt"), script);
    descriptor.putString(charIDToTypeID("jsMs"), script.name);
    number(0xffffffff,4); parts.push(descriptor.toStream().substring(4));
    var actionFile = new File(work.fsName + "/Chinminator-ImageProcessor.atn");
    actionFile.encoding = "BINARY";
    if (!actionFile.open("w")) throw new Error("Cannot write Action file");
    actionFile.write(parts.join("")); actionFile.close();
    app.load(actionFile);
    alert("Installed: Chinminator Image Processor / Retouch. Select it under Run Action in Image Processor.");
})();
