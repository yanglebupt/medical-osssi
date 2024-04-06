import { lazy } from "react";
import { isMobile } from "react-device-detect";

export default lazy(async () => import(isMobile ? "./h5/index" : "./pc/index"));
