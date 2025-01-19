import { isMobile } from "react-device-detect";
import { lazy, Suspense, useEffect, useState } from "react";

const H5 = lazy(() => import("./app/h5/index"));
const PC = lazy(() => import("./app/pc/index"));

const H5_GLOBAL_CSS = ()=>import("./global.h5.css");
const PC_GLOBAL_CSS = ()=>import("./global.css");

export default ()=>{
    const [show, setShow] = useState(false)
    useEffect(()=>{
        isMobile ? H5_GLOBAL_CSS(): PC_GLOBAL_CSS()
        setShow(true)
    })
    return show && <Suspense>
    {isMobile ? <H5 /> : <PC/>}
    </Suspense>
}
