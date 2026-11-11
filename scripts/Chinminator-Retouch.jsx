#target photoshop
// Install the native Action using Install-ImageProcessor-Action.jsx. Keep this file in scripts/.
(function () {
    if (!app.documents.length) throw new Error("Chinminator: no active document");
    var source = app.activeDocument;
    if (source.mode !== DocumentMode.RGB || source.bitsPerChannel !== BitsPerChannelType.EIGHT)
        throw new Error("Chinminator requires RGB, 8 bits/channel");
    var root = File($.fileName).parent.parent;
    var config = new File(root.fsName + "/work/automation-settings.json");
    var heartbeat = new File(root.fsName + "/work/automation-heartbeat.txt");
    if (!heartbeat.exists || new Date().getTime() - heartbeat.modified.getTime() > 15000)
        throw new Error("Chinminator server is not running. Run Start from this project folder.");
    if (!config.exists)
        throw new Error("Chinminator: first save Image Processor settings from the plugin panel.");
    var jobs = new Folder(root.fsName + "/work/automation-jobs");
    if (!jobs.exists && !jobs.create()) throw new Error("Cannot create Chinminator job folder");
    var job = new Folder(jobs.fsName + "/" + new Date().getTime() + "-" + Math.floor(Math.random()*1000000000));
    if (!job.create()) throw new Error("Cannot create Chinminator job");
    if (!config.copy(job.fsName + "/settings.json")) throw new Error("Cannot copy retouch settings");
    var previousDialogs = app.displayDialogs;
    var temporary = null, result = null;
    function text(file) {
        file.encoding = "UTF8";
        if (!file.open("r")) throw new Error("Cannot read " + file.fsName);
        var value = file.read(); file.close(); return value;
    }
    try {
        app.displayDialogs = DialogModes.NO;
        temporary = source.duplicate("__Chinminator_input__", true);
        temporary.saveAs(new File(job.fsName + "/input.png"), new PNGSaveOptions(), true);
        temporary.close(SaveOptions.DONOTSAVECHANGES); temporary = null;
        app.activeDocument = source;
        var documentReference = new ActionReference();
        documentReference.putProperty(charIDToTypeID("Prpr"), charIDToTypeID("DocI"));
        documentReference.putEnumerated(charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        var documentId = executeActionGet(documentReference).getInteger(charIDToTypeID("DocI"));
        var scope = new File(job.fsName + "/document-id.txt");
        if (!scope.open("w")) throw new Error("Cannot write document scope");
        scope.write(String(documentId)); scope.close();
        var request = new File(job.fsName + "/request.pending");
        if (!request.open("w")) throw new Error("Cannot write Chinminator request");
        request.write("ready"); request.close();
        if (!request.rename("request.txt")) throw new Error("Cannot submit Chinminator request");
        var started = new Date().getTime();
        while (true) {
            var failure = new File(job.fsName + "/error.txt");
            if (failure.exists) throw new Error("Chinminator: " + text(failure));
            var done = new File(job.fsName + "/done.txt");
            if (done.exists) {
                if (text(done) === "skipped") return;
                break;
            }
            if (new Date().getTime() - started > 1800000)
                throw new Error("Chinminator timed out after 30 minutes. Job: " + job.fsName);
            var alive = new File(root.fsName + "/work/automation-heartbeat.txt");
            if (!alive.exists || new Date().getTime() - alive.modified.getTime() > 15000)
                throw new Error("Chinminator server stopped while processing. Restart Start and run again.");
            $.sleep(200);
        }
        result = app.open(new File(job.fsName + "/output.png"));
        if (result.width.as("px") !== source.width.as("px") || result.height.as("px") !== source.height.as("px"))
            throw new Error("Chinminator result dimensions changed");
        var layer = result.activeLayer.duplicate(source, ElementPlacement.PLACEATBEGINNING);
        app.activeDocument = source;
        layer.name = "Chinminator Fixed";
        app.activeDocument = result;
        result.close(SaveOptions.DONOTSAVECHANGES); result = null;
        app.activeDocument = source;
    } finally {
        if (temporary) temporary.close(SaveOptions.DONOTSAVECHANGES);
        if (result) result.close(SaveOptions.DONOTSAVECHANGES);
        app.activeDocument = source;
        app.displayDialogs = previousDialogs;
    }
})();
